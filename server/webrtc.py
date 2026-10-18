###############################################################################
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#  email: lipku@foxmail.com
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################

import asyncio
import json
import logging
import threading
import time
from typing import Tuple, Dict, Optional, Set, Union
import queue
from av.frame import Frame
from av.packet import Packet
from av import AudioFrame
import fractions
import numpy as np

AUDIO_PTIME = 0.020  # 20ms audio packetization
VIDEO_CLOCK_RATE = 90000
VIDEO_PTIME = 0.040 #1 / 25  # 30fps
VIDEO_TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)
SAMPLE_RATE = 16000
AUDIO_TIME_BASE = fractions.Fraction(1, SAMPLE_RATE)

#from aiortc.contrib.media import MediaPlayer, MediaRelay
#from aiortc.rtcrtpsender import RTCRtpSender
from aiortc import (
    MediaStreamTrack,
)

logging.basicConfig()
logger = logging.getLogger(__name__)
from utils.logger import logger as mylogger


class PlayerStreamTrack(MediaStreamTrack):
    """
    A video track that returns an animated flag.
    """

    def __init__(self, player, kind):
        super().__init__()  # don't forget this!
        self.kind = kind
        self._player = player
        queue_size = (25 if kind == 'video' else 50) if player._drop_stale_media else 100
        self._queue = queue.Queue(maxsize=queue_size)
        self.timelist = [] #记录最近包的时间戳
        self.current_frame_count = 0
        if self.kind == 'video':
            self.framecount = 0
            self.lasttime = time.perf_counter()
            self.totaltime = 0
    
    _start: float
    _timestamp: int

    async def next_timestamp(self) -> Tuple[int, fractions.Fraction]:
        if self.readyState != "live":
            raise Exception

        if self.kind == 'video':
            if hasattr(self, "_timestamp"):
                #self._timestamp = (time.time()-self._start) * VIDEO_CLOCK_RATE
                self._timestamp += int(VIDEO_PTIME * VIDEO_CLOCK_RATE)
                self.current_frame_count += 1
                wait = self._start + self.current_frame_count * VIDEO_PTIME - time.time()
                # wait = self.timelist[0] + len(self.timelist)*VIDEO_PTIME - time.time()               
                if wait>0:
                    await asyncio.sleep(wait)
                # if len(self.timelist)>=100:
                #     self.timelist.pop(0)
                # self.timelist.append(time.time())
            else:
                if self._player._synchronize_tracks:
                    if self._player._media_start_time is None:
                        self._player._media_start_time = time.time()
                    elapsed = max(0, time.time() - self._player._media_start_time)
                    self._start = self._player._media_start_time
                    self.current_frame_count = round(elapsed / VIDEO_PTIME)
                    self._timestamp = self.current_frame_count * int(VIDEO_PTIME * VIDEO_CLOCK_RATE)
                else:
                    self._start = time.time()
                    self._timestamp = 0
                self.timelist.append(self._start)
                mylogger.info('%s start:%f', self.kind, self._start)
            return self._timestamp, VIDEO_TIME_BASE
        else: #audio
            if hasattr(self, "_timestamp"):
                #self._timestamp = (time.time()-self._start) * SAMPLE_RATE
                self._timestamp += int(AUDIO_PTIME * SAMPLE_RATE)
                self.current_frame_count += 1
                wait = self._start + self.current_frame_count * AUDIO_PTIME - time.time()
                # wait = self.timelist[0] + len(self.timelist)*AUDIO_PTIME - time.time()
                if wait>0:
                    await asyncio.sleep(wait)
                # if len(self.timelist)>=200:
                #     self.timelist.pop(0)
                #     self.timelist.pop(0)
                # self.timelist.append(time.time())
            else:
                if self._player._synchronize_tracks:
                    if self._player._media_start_time is None:
                        self._player._media_start_time = time.time()
                    elapsed = max(0, time.time() - self._player._media_start_time)
                    self._start = self._player._media_start_time
                    self.current_frame_count = round(elapsed / AUDIO_PTIME)
                    self._timestamp = self.current_frame_count * int(AUDIO_PTIME * SAMPLE_RATE)
                else:
                    self._start = time.time()
                    self._timestamp = 0
                self.timelist.append(self._start)
                mylogger.info('%s start:%f', self.kind, self._start)
            return self._timestamp, AUDIO_TIME_BASE

    async def recv(self) -> Union[Frame, Packet]:
        # frame = self.frames[self.counter % 30]            
        self._player._start(self)
        # if self.kind == 'video':
        #     frame = await self._queue.get()
        # else: #audio
        #     if hasattr(self, "_timestamp"):
        #         wait = self._start + self._timestamp / SAMPLE_RATE + AUDIO_PTIME - time.time()
        #         if wait>0:
        #             await asyncio.sleep(wait)
        #         if self._queue.qsize()<1:
        #             #frame = AudioFrame(format='s16', layout='mono', samples=320)
        #             audio = np.zeros((1, 320), dtype=np.int16)
        #             frame = AudioFrame.from_ndarray(audio, layout='mono', format='s16')
        #             frame.sample_rate=16000
        #         else:
        #             frame = await self._queue.get()
        #     else:
        while True:
            try:
                item = self._queue.get_nowait()
                break
            except queue.Empty:
                await asyncio.sleep(0.005)
        if len(item) == 4:
            frame, eventpoint, pts, time_base = item
            target_time = self._player._media_start_time + float(pts * time_base)
            wait = target_time - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
        else:
            frame, eventpoint = item
            pts, time_base = await self.next_timestamp()
        frame.pts = pts
        frame.time_base = time_base
        if eventpoint and self._player is not None:
            self._player.notify(eventpoint)
        if frame is None:
            self.stop()
            raise Exception
        if self.kind == 'video':
            self.totaltime += (time.perf_counter() - self.lasttime)
            self.framecount += 1
            self.lasttime = time.perf_counter()
            if self.framecount==100:
                mylogger.info(f"------actual avg final fps:{self.framecount/self.totaltime:.4f}")
                self.framecount = 0
                self.totaltime=0
        return frame
    
    def stop(self):
        super().stop()
        # Drain & delete remaining frames
        while not self._queue.empty():
            item = self._queue.get_nowait()
            del item
        if self._player is not None:
            self._player._stop(self)
            self._player = None

def player_worker_thread(
    quit_event,
    container
):
    container.render(quit_event)

class HumanPlayer:

    def __init__(
        self, avatar_session, format=None, options=None, timeout=None, loop=False, decode=True
    ):
        self.__thread: Optional[threading.Thread] = None
        self.__thread_quit: Optional[threading.Event] = None

        # examine streams
        self.__started: Set[PlayerStreamTrack] = set()
        self.__audio: Optional[PlayerStreamTrack] = None
        self.__video: Optional[PlayerStreamTrack] = None
        self._synchronize_tracks = bool(
            getattr(avatar_session, 'is_streaming_avatar', False)
        )
        self._drop_stale_media = self._synchronize_tracks
        self._media_start_time = None
        self._media_audio_pts = 0
        self._media_video_pts = 0
        self._media_last_push_time = None
        self._media_bundle_lock = threading.RLock()

        self.__audio = PlayerStreamTrack(self, kind="audio")
        self.__video = PlayerStreamTrack(self, kind="video")

        self.__container = avatar_session
        if hasattr(self.__container, 'output'):
            self.__container.output._player = self

    def push_video(self, frame):
        from av import VideoFrame
        new_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        with self._media_bundle_lock:
            if self._synchronize_tracks:
                pts, time_base = self._next_media_pts("video", 3600)
                self._enqueue_media(self.__video, (new_frame, None, pts, time_base))
            else:
                self._enqueue_media(self.__video, (new_frame, None))

    def push_audio(self, frame, eventpoint=None):
        from av import AudioFrame
        new_frame = AudioFrame(format='s16', layout='mono', samples=frame.shape[0])
        new_frame.planes[0].update(frame.tobytes())
        new_frame.sample_rate = 16000
        with self._media_bundle_lock:
            if self._synchronize_tracks:
                pts, time_base = self._next_media_pts("audio", new_frame.samples)
                self._enqueue_media(
                    self.__audio, (new_frame, eventpoint, pts, time_base)
                )
            else:
                self._enqueue_media(self.__audio, (new_frame, eventpoint))

    def push_media_bundle(self, audio_frame, video_frame=None, eventpoint=None):
        from av import AudioFrame, VideoFrame

        new_audio = AudioFrame(
            format='s16', layout='mono', samples=audio_frame.shape[0]
        )
        new_audio.planes[0].update(audio_frame.tobytes())
        new_audio.sample_rate = SAMPLE_RATE
        new_video = (
            VideoFrame.from_ndarray(video_frame, format="bgr24")
            if video_frame is not None
            else None
        )
        with self._media_bundle_lock:
            if not self._synchronize_tracks:
                self._enqueue_media(self.__audio, (new_audio, eventpoint))
                if new_video is not None:
                    self._enqueue_media(self.__video, (new_video, None))
                return

            self._advance_media_clock()
            audio_pts = self._media_audio_pts
            self._media_audio_pts += new_audio.samples
            self._enqueue_media(
                self.__audio,
                (new_audio, eventpoint, audio_pts, AUDIO_TIME_BASE),
            )
            if new_video is not None:
                video_pts = self._media_video_pts
                self._media_video_pts += 3600
                self._enqueue_media(
                    self.__video,
                    (new_video, None, video_pts, VIDEO_TIME_BASE),
                )

    def _advance_media_clock(self):
        now = time.monotonic()
        if self._media_start_time is None:
            self._media_start_time = now
        if self._media_last_push_time is not None:
            elapsed = now - self._media_last_push_time
            if elapsed > AUDIO_PTIME * 3:
                self._media_start_time += elapsed - AUDIO_PTIME
        self._media_last_push_time = now

    def _next_media_pts(self, kind, duration):
        self._advance_media_clock()
        if kind == "audio":
            pts = self._media_audio_pts
            self._media_audio_pts += duration
            return pts, AUDIO_TIME_BASE
        pts = self._media_video_pts
        self._media_video_pts += duration
        return pts, VIDEO_TIME_BASE

    def _enqueue_media(self, track, item):
        if not self._drop_stale_media:
            track._queue.put(item)
            return
        try:
            track._queue.put_nowait(item)
        except queue.Full:
            try:
                track._queue.get_nowait()
            except queue.Empty:
                pass
            track._queue.put_nowait(item)

    def get_buffer_size(self) -> int:
        return self.__video._queue.qsize()

    def notify(self,eventpoint):
        if self.__container is not None:
            self.__container.notify(eventpoint)

    @property
    def audio(self) -> MediaStreamTrack:
        """
        A :class:`aiortc.MediaStreamTrack` instance if the file contains audio.
        """
        return self.__audio

    @property
    def video(self) -> MediaStreamTrack:
        """
        A :class:`aiortc.MediaStreamTrack` instance if the file contains video.
        """
        return self.__video

    def _start(self, track: PlayerStreamTrack) -> None:
        self.__started.add(track)
        if self.__thread is None:
            self.__log_debug("Starting worker thread")
            self.__thread_quit = threading.Event()
            self.__thread = threading.Thread(
                name="media-player",
                target=player_worker_thread,
                args=(
                    self.__thread_quit,
                    self.__container
                ),
            )
            self.__thread.start()

    def _stop(self, track: PlayerStreamTrack) -> None:
        self.__started.discard(track)

        if not self.__started and self.__thread is not None:
            self.__log_debug("Stopping worker thread")
            self.__thread_quit.set()
            self.__thread.join()
            self.__thread = None

        if not self.__started and self.__container is not None:
            #self.__container.close()
            self.__container = None

    def __log_debug(self, msg: str, *args) -> None:
        mylogger.debug(f"HumanPlayer {msg}", *args)
