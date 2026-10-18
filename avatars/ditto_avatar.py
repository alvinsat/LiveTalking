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

    def __init__(self, opt, model=None, avatar=None):
        super().__init__(opt)
        self.audio_queue = queue.Queue(maxsize=250)
        self.video_queue = queue.Queue(maxsize=8)
        self._frame_lock = threading.Lock()
        self._latest_frame = cv2.imread(opt.ditto_source_path)
        if self._latest_frame is None:
            source_video = cv2.VideoCapture(opt.ditto_source_path)
            ok, self._latest_frame = source_video.read()
            source_video.release()
        if self._latest_frame is None:
            raise ValueError(f"Could not read Ditto source: {opt.ditto_source_path}")

        self._connection = socket.create_connection(
            (opt.ditto_worker_host, opt.ditto_worker_port), timeout=10
        )
        self._connection.sendall(b"\x02")
        self._connection.settimeout(None)
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
                if length <= 0 or length > 16 * 1024 * 1024:
                    raise RuntimeError(f"Invalid Ditto frame size: {length}")
                payload = self._recv_exact(self._connection, length)
                frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise RuntimeError("Could not decode a frame from the Ditto worker")
                try:
                    self.video_queue.put_nowait(frame)
                except queue.Full:
                    try:
                        self.video_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.video_queue.put_nowait(frame)
        except (OSError, ConnectionError, RuntimeError, cv2.error) as error:
            if not getattr(self, "_closing", False):
                self._reader_error = error

    def render(self, quit_event):
        self.quit_event = quit_event
        self.init_customindex()
        self.tts.render(quit_event)
        self.output.start()

        next_audio_time = time.monotonic()
        tick = 0
        self._closing = False
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

                pcm16 = np.clip(audio_chunk * 32767.0, -32768, 32767).astype("<i2")
                self._connection.sendall(pcm16.tobytes())
                self.output.push_audio_frame(pcm16, eventpoint)
                self.record_audio_data(pcm16)
                self.speaking = bool(np.any(audio_chunk))

                if tick % 2 == 0:
                    while True:
                        try:
                            frame = self.video_queue.get_nowait()
                        except queue.Empty:
                            break
                        with self._frame_lock:
                            self._latest_frame = frame
                    with self._frame_lock:
                        frame = self._latest_frame.copy()
                    self.output.push_video_frame(frame)
                    self.record_video_data(frame)

                tick += 1
                next_audio_time += 0.02
                if next_audio_time < time.monotonic() - 0.1:
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
