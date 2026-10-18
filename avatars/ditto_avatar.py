from collections import deque
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


@register("avatar", "ditto")
class DittoAvatar(BaseAvatar):
    is_streaming_avatar = True
    speech_hangover_seconds = 0.4

    def __init__(self, opt, model=None, avatar=None):
        super().__init__(opt)
        self.audio_queue = queue.Queue(maxsize=50)
        self.video_queue = queue.Queue(maxsize=8)
        self._pending_audio = deque()
        self._audio_frame_credit = 0
        self._frame_lock = threading.Lock()
        self._latest_frame = cv2.imread(opt.ditto_source_path)
        if self._latest_frame is None:
            source_video = cv2.VideoCapture(opt.ditto_source_path)
            ok, self._latest_frame = source_video.read()
            source_video.release()
        if self._latest_frame is None:
            raise ValueError(f"Could not read Ditto source: {opt.ditto_source_path}")
        self._idle_frame = self._latest_frame.copy()

        self._connection = socket.create_connection(
            (opt.ditto_worker_host, opt.ditto_worker_port), timeout=10
        )
        self._connection.sendall(b"\x02")
        height, width = struct.unpack("!II", self._recv_exact(self._connection, 8))
        if not height or not width:
            self._connection.close()
            raise RuntimeError("Ditto worker returned an invalid frame size")
        self._frame_shape = (height, width, 3)
        if self._idle_frame.shape[:2] != (height, width):
            self._idle_frame = cv2.resize(self._idle_frame, (width, height))
            self._latest_frame = self._idle_frame.copy()
        self._connection.settimeout(None)
        self._closing = False
        self._reader_error = None
        self._reader_thread = threading.Thread(
            target=self._read_video, daemon=True, name="ditto-video-reader"
        )
        self._reader_thread.start()

    def put_audio_frame(self, audio_chunk, datainfo=None):
        audio_chunk = np.asarray(audio_chunk, dtype=np.float32).reshape(-1)
        if len(audio_chunk) != self.chunk:
            raise ValueError(
                f"Ditto expects {self.chunk}-sample audio packets, received {len(audio_chunk)}"
            )
        self.audio_queue.put((audio_chunk.copy(), datainfo or {}))

    def flush_talk(self):
        super().flush_talk()
        while True:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
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
                expected_bytes = int(np.prod(self._frame_shape))
                if length != expected_bytes:
                    raise RuntimeError(f"Invalid Ditto frame size: {length}")
                payload = self._recv_exact(self._connection, length)
                frame_rgb = np.frombuffer(payload, dtype=np.uint8).reshape(self._frame_shape)
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                while not self._closing:
                    try:
                        self.video_queue.put(frame, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        except (OSError, ConnectionError, RuntimeError, cv2.error) as error:
            if not getattr(self, "_closing", False):
                self._reader_error = error

    def render(self, quit_event):
        self.quit_event = quit_event
        self.init_customindex()
        self.tts.render(quit_event)
        self.output.start()

        next_audio_time = time.monotonic()
        last_input_speech_time = None
        last_output_speech_time = None
        tick = 0
        logger.info("Starting persistent Ditto avatar stream for session %s", self.sessionid)
        try:
            while not quit_event.is_set():
                delay = next_audio_time - time.monotonic()
                if delay > 0:
                    quit_event.wait(delay)
                    if quit_event.is_set():
                        break

                try:
                    audio_chunk, eventpoint = self.audio_queue.get_nowait()
                except queue.Empty:
                    audio_chunk = np.zeros((self.chunk,), dtype=np.float32)
                    eventpoint = {}

                now = time.monotonic()
                if float(np.sqrt(np.mean(audio_chunk * audio_chunk))) >= 0.003:
                    last_input_speech_time = now
                input_active = (
                    last_input_speech_time is not None
                    and now - last_input_speech_time <= self.speech_hangover_seconds
                )
                if input_active:
                    self._pending_audio.append((audio_chunk.copy(), eventpoint))

                pcm16 = np.clip(audio_chunk * 32767.0, -32768, 32767).astype("<i2")
                self._connection.sendall(pcm16.tobytes())

                if tick % 2 == 0:
                    try:
                        frame = self.video_queue.get_nowait()
                    except queue.Empty:
                        frame = None
                    if frame is not None:
                        with self._frame_lock:
                            self._latest_frame = frame
                        self._audio_frame_credit += 2

                if self._audio_frame_credit:
                    self._audio_frame_credit -= 1
                    try:
                        audio_chunk, eventpoint = self._pending_audio.popleft()
                    except IndexError:
                        audio_chunk = np.zeros((self.chunk,), dtype=np.float32)
                        eventpoint = {}
                else:
                    audio_chunk = np.zeros((self.chunk,), dtype=np.float32)
                    eventpoint = {}

                pcm16 = np.clip(audio_chunk * 32767.0, -32768, 32767).astype("<i2")
                self.output.push_audio_frame(pcm16, eventpoint)
                self.record_audio_data(pcm16)
                output_now = time.monotonic()
                if float(np.sqrt(np.mean(audio_chunk * audio_chunk))) >= 0.003:
                    last_output_speech_time = output_now
                self.speaking = (
                    last_output_speech_time is not None
                    and output_now - last_output_speech_time <= self.speech_hangover_seconds
                )

                if tick % 2 == 0:
                    if self.speaking:
                        with self._frame_lock:
                            frame = self._latest_frame.copy()
                    else:
                        frame = self._idle_frame.copy()
                    self.output.push_video_frame(frame)
                    self.record_video_data(frame)

                tick += 1
                next_audio_time += 0.02
                if next_audio_time < time.monotonic():
                    next_audio_time = time.monotonic()

                if self._reader_error is not None:
                    raise RuntimeError("Ditto worker video stream failed") from self._reader_error
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
