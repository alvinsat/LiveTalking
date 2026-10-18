import queue
import socket
import struct
import threading
import time

import cv2
import numpy as np

from avatars.base_avatar import BaseAvatar
from registry import register
from utils.logger import logger


_END_RESPONSE = object()


@register("avatar", "ditto")
class DittoAvatar(BaseAvatar):
    is_streaming_avatar = True
    idle_video_fps = 12.5
    max_cached_idle_frames = 64

    def __init__(self, opt, model=None, avatar=None):
        super().__init__(opt)
        self.audio_queue = queue.Queue(maxsize=50)
        self.video_queue = queue.Queue()
        self._interrupt_requested = threading.Event()
        idle_path = opt.ditto_idle_video_path or opt.ditto_source_path
        self._latest_frame = cv2.imread(idle_path)
        self._idle_video_path = self._latest_frame is None
        if self._latest_frame is None:
            source_video = cv2.VideoCapture(idle_path)
            ok, self._latest_frame = source_video.read()
            source_video.release()
        if self._latest_frame is None:
            raise ValueError(f"Could not read Ditto idle source: {idle_path}")
        self._idle_frame = self._latest_frame.copy()

        self._connection = socket.create_connection(
            (opt.ditto_worker_host, opt.ditto_worker_port), timeout=10
        )
        self._connection.sendall(b"\x02")
        height, width = struct.unpack("!II", self._recv_exact(self._connection, 8))
        if not height or not width:
            self._connection.close()
            raise RuntimeError("Ditto worker returned an invalid frame size")
        if self._idle_frame.shape[:2] != (height, width):
            self._idle_frame = cv2.resize(self._idle_frame, (width, height))
            self._latest_frame = self._idle_frame.copy()
        if self._idle_video_path:
            self._idle_frames, self._idle_video_fps = self._cache_idle_video(
                idle_path, width, height
            )
        else:
            self._idle_frames = [self._encode_idle_frame(self._idle_frame)]
            self._idle_video_fps = self.idle_video_fps
        self._idle_frame_index = 0
        self._idle_frame = self._decode_idle_frame(0)
        self._connection.settimeout(None)
        self._closing = False
        self._reader_error = None
        self._reader_thread = threading.Thread(
            target=self._read_video, daemon=True, name="ditto-video-reader"
        )
        self._reader_thread.start()

    @staticmethod
    def _encode_idle_frame(frame):
        ok, encoded = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85]
        )
        if not ok:
            raise RuntimeError("Could not cache a Ditto idle frame")
        return encoded

    def _cache_idle_video(self, path, width, height):
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"Could not reopen Ditto idle video: {path}")

        source_fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(source_fps) or source_fps <= 0:
            source_fps = 25.0
        sample_every = max(1, round(source_fps / self.idle_video_fps))
        cached_fps = source_fps / sample_every
        frames = []
        frame_index = 0
        try:
            while len(frames) < self.max_cached_idle_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index % sample_every == 0:
                    if frame.shape[:2] != (height, width):
                        frame = cv2.resize(frame, (width, height))
                    frames.append(self._encode_idle_frame(frame))
                frame_index += 1
        finally:
            capture.release()

        if not frames:
            raise ValueError(f"Ditto idle video contains no readable frames: {path}")
        logger.info(
            "Cached %d Ditto idle frames at %.2f fps",
            len(frames),
            cached_fps,
        )
        return frames, cached_fps

    def _decode_idle_frame(self, index):
        frame = cv2.imdecode(self._idle_frames[index], cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Could not decode cached Ditto idle frame {index}")
        return frame

    def _get_idle_frame(self, now, loop_start):
        if len(self._idle_frames) == 1:
            return self._idle_frame
        index = int((now - loop_start) * self._idle_video_fps) % len(self._idle_frames)
        if index != self._idle_frame_index:
            self._idle_frame = self._decode_idle_frame(index)
            self._idle_frame_index = index
        return self._idle_frame

    def put_audio_frame(self, audio_chunk, datainfo=None):
        audio_chunk = np.asarray(audio_chunk, dtype=np.float32).reshape(-1)
        if len(audio_chunk) != self.chunk:
            raise ValueError(
                f"Ditto expects {self.chunk}-sample audio packets, received {len(audio_chunk)}"
            )
        self.audio_queue.put((audio_chunk.copy(), dict(datainfo or {})))

    def flush_talk(self):
        super().flush_talk()
        while True:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        self._interrupt_requested.set()
        self.speaking = False

    @staticmethod
    def _recv_exact(connection, size):
        data = bytearray()
        while len(data) < size:
            part = connection.recv(size - len(data))
            if not part:
                raise ConnectionError("Ditto worker closed the video connection")
            data.extend(part)
        return bytes(data)

    def _read_video(self):
        try:
            while True:
                header = self._recv_exact(self._connection, 4)
                (length,) = struct.unpack("!I", header)
                if length == 0:
                    raise RuntimeError("Ditto worker stopped its video stream")
                if length == 0xFFFFFFFF:
                    self.video_queue.put(_END_RESPONSE)
                    continue
                payload = self._recv_exact(self._connection, length)
                self.video_queue.put(payload)
        except (OSError, ConnectionError, RuntimeError, cv2.error) as error:
            if not getattr(self, "_closing", False):
                self._reader_error = error

    def render(self, quit_event):
        self.quit_event = quit_event
        self.init_customindex()
        self.tts.render(quit_event)
        self.output.start()

        next_audio_time = time.monotonic()
        idle_loop_start = next_audio_time
        response_audio = []
        response_frames = []
        response_started = False
        response_end_sent = False
        discard_response = False
        playback_audio = None
        playback_frames = None
        playback_index = 0
        last_audio_received = None
        tick = 0
        logger.info("Starting persistent Ditto avatar stream for session %s", self.sessionid)
        try:
            while not quit_event.is_set():
                delay = next_audio_time - time.monotonic()
                if delay > 0:
                    quit_event.wait(delay)
                    if quit_event.is_set():
                        break

                if self._interrupt_requested.is_set():
                    self._interrupt_requested.clear()
                    if playback_audio is not None:
                        response_started = False
                        response_end_sent = False
                        discard_response = False
                    elif response_started:
                        if not response_end_sent:
                            self._connection.sendall(b"\x02")
                            response_end_sent = True
                        discard_response = True
                    else:
                        discard_response = False
                    playback_audio = None
                    playback_frames = None
                    playback_index = 0
                    response_audio.clear()
                    response_frames.clear()
                    self.speaking = False

                if playback_audio is not None:
                    audio_chunk, eventpoint = playback_audio[playback_index]
                    pcm16 = np.clip(audio_chunk * 32767.0, -32768, 32767).astype("<i2")
                    video_frame = None
                    if playback_index % 2 == 0 and playback_frames:
                        video_slot = playback_index // 2
                        output_video_slots = (len(playback_audio) + 1) // 2
                        frame_index = min(
                            video_slot * len(playback_frames) // output_video_slots,
                            len(playback_frames) - 1,
                        )
                        encoded = np.frombuffer(playback_frames[frame_index], dtype=np.uint8)
                        video_frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                        if video_frame is None:
                            raise RuntimeError("Could not decode a generated Ditto frame")
                    self.output.push_media_bundle(pcm16, video_frame, eventpoint)
                    self.record_audio_data(pcm16)
                    if video_frame is not None:
                        self.record_video_data(video_frame)
                    self.speaking = (
                        float(np.sqrt(np.mean(audio_chunk * audio_chunk))) >= 0.003
                    )
                    playback_index += 1
                    if playback_index >= len(playback_audio):
                        playback_audio = None
                        playback_frames = None
                        playback_index = 0
                        response_audio.clear()
                        response_frames.clear()
                        response_started = False
                        response_end_sent = False
                        discard_response = False
                        self.speaking = False
                else:
                    if response_started and not response_end_sent:
                        try:
                            audio_chunk, eventpoint = self.audio_queue.get_nowait()
                        except queue.Empty:
                            audio_chunk = None
                            eventpoint = {}
                        if audio_chunk is not None:
                            response_audio.append((audio_chunk, eventpoint))
                            self._connection.sendall(
                                b"\x01"
                                + np.clip(
                                    audio_chunk * 32767.0, -32768, 32767
                                ).astype("<i2").tobytes()
                            )
                            last_audio_received = time.monotonic()
                            if eventpoint.get("status") == "end":
                                self._connection.sendall(b"\x02")
                                response_end_sent = True
                        elif (
                            last_audio_received is not None
                            and time.monotonic() - last_audio_received >= 1.0
                        ):
                            self._connection.sendall(b"\x02")
                            response_end_sent = True
                            logger.warning(
                                "Ditto ended an audio response after 1 second without more audio"
                            )

                    elif not response_started:
                        try:
                            audio_chunk, eventpoint = self.audio_queue.get_nowait()
                        except queue.Empty:
                            audio_chunk = None
                            eventpoint = {}
                        if audio_chunk is not None:
                            response_started = True
                            response_audio.append((audio_chunk, eventpoint))
                            self._connection.sendall(
                                b"\x01"
                                + np.clip(
                                    audio_chunk * 32767.0, -32768, 32767
                                ).astype("<i2").tobytes()
                            )
                            last_audio_received = time.monotonic()
                            if eventpoint.get("status") == "end":
                                self._connection.sendall(b"\x02")
                                response_end_sent = True

                    while True:
                        try:
                            generated = self.video_queue.get_nowait()
                        except queue.Empty:
                            break
                        if generated is _END_RESPONSE:
                            if discard_response:
                                response_audio.clear()
                                response_frames.clear()
                                response_started = False
                                response_end_sent = False
                                discard_response = False
                            elif response_audio:
                                if not response_frames:
                                    logger.error(
                                        "Ditto completed an utterance without generating video; response discarded"
                                    )
                                    response_audio.clear()
                                    response_started = False
                                    response_end_sent = False
                                else:
                                    playback_audio = response_audio
                                    playback_frames = response_frames
                                    playback_index = 0
                                    logger.info(
                                        "Ditto response ready: %d audio packets, %d generated frames",
                                        len(playback_audio),
                                        len(playback_frames),
                                    )
                            break
                        if not discard_response:
                            response_frames.append(generated)

                    pcm16 = np.zeros((self.chunk,), dtype="<i2")
                    video_frame = None
                    if tick % 2 == 0:
                        video_frame = self._get_idle_frame(
                            time.monotonic(), idle_loop_start
                        )
                    self.output.push_media_bundle(pcm16, video_frame)
                    self.record_audio_data(pcm16)
                    if video_frame is not None:
                        self.record_video_data(video_frame)

                tick += 1
                next_audio_time += 0.02
                if next_audio_time < time.monotonic():
                    next_audio_time = time.monotonic()

                if self._reader_error is not None:
                    raise RuntimeError("Ditto worker video stream failed") from self._reader_error
        except (ConnectionResetError, BrokenPipeError) as error:
            if not quit_event.is_set():
                logger.warning("Ditto worker connection closed: %s", error)
        except (OSError, RuntimeError):
            if not quit_event.is_set():
                logger.exception("Ditto avatar stream failed")
        finally:
            self._closing = True
            try:
                self._connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._connection.close()
            self._reader_thread.join(timeout=1)
            self.output.stop()
            logger.info("Ditto avatar stream stopped for session %s", self.sessionid)
