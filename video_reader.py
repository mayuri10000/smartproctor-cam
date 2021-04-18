import threading
from typing import Optional, Set

import av
from av import AudioFrame, VideoFrame
import asyncio
import time
from aiortc.mediastreams import MediaStreamTrack, MediaStreamError

# This is the FIFO pipe that outputs the raw frames of the camera capture
DEEPLENS_VIDEO_FIFO_NAME = "/opt/awscam/out/ch1_out.h264"


class DeepLensVideoTrack(MediaStreamTrack):
    def __init__(self, player, kind):
        super().__init__()
        self.kind = kind
        self._player = player
        self.queue = asyncio.Queue()
        self.start = None

    async def recv(self):
        if self.readyState != "live":
            raise MediaStreamError

        self._player.start(self)
        frame = await self.queue.get()
        if frame is None:
            self.stop()
            raise MediaStreamError
        frame_time = frame.time

        # control playback rate
        if (
            self._player is not None
            and frame_time is not None
        ):
            if self.start is None:
                self.start = time.time() - frame_time
            else:
                wait = self.start + frame_time - time.time()
                await asyncio.sleep(wait)

        return frame

    def stop(self):
        super().stop()
        if self._player is not None:
            self._player.stop(self)
            self._player = None


def player_worker(
    loop, container, streams, video_track, quit_event
):
    while not quit_event.is_set():
        try:
            frame = next(container.decode(*streams))
        except (av.AVError, StopIteration):
            if video_track:
                asyncio.run_coroutine_threadsafe(video_track.queue.put(None), loop)
            break
        # we should decode and send every key frames
        # and also make sure there is not to much frames in the queue
        # as this will lead to memory leak
        if frame.key_frame == 1 or video_track.queue.qsize() <= 1:
            frame.pts = frame.index * 48
            asyncio.run_coroutine_threadsafe(video_track.queue.put(frame), loop)


class DeepLensVideoReader:
    """
    Reads video frames from DeepLens camera capture
    """

    def __init__(self):
        self.__container = av.open(file=DEEPLENS_VIDEO_FIFO_NAME, mode="r")
        self.__thread: Optional[threading.Thread] = None
        self.__thread_quit: Optional[threading.Event] = None

        # examine streams
        self.__started: Set[DeepLensVideoTrack] = set()
        self.__streams = []
        self.__video: Optional[DeepLensVideoTrack] = None
        for stream in self.__container.streams:
            if stream.type == "video" and not self.__video:
                self.__video = DeepLensVideoTrack(self, kind="video")
                self.__streams.append(stream)

    @property
    def video(self) -> MediaStreamTrack:
        """
        A :class:`aiortc.MediaStreamTrack` instance if the file contains video.
        """
        return self.__video

    def start(self, track: DeepLensVideoTrack) -> None:
        self.__started.add(track)
        if self.__thread is None:
            self.__thread_quit = threading.Event()
            self.__thread = threading.Thread(
                name="media-player",
                target=player_worker,
                args=(
                    asyncio.get_event_loop(),
                    self.__container,
                    self.__streams,
                    self.__video,
                    self.__thread_quit
                ),
            )
            self.__thread.start()

    def stop(self, track: DeepLensVideoTrack) -> None:
        self.__started.discard(track)

        if not self.__started and self.__thread is not None:
            self.__thread_quit.set()
            self.__thread.join()
            self.__thread = None

        if not self.__started and self.__container is not None:
            self.__container.close()
            self.__container = None


