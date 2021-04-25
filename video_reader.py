import fractions
import os
import stat
import threading
from typing import Optional, Set

import numpy as np
from threading import Thread, Event
import queue
import av
from av import AudioFrame, VideoFrame
import asyncio
import time
import awscam
import cv2
from aiortc.mediastreams import MediaStreamTrack, MediaStreamError, VideoStreamTrack
from aiortc.contrib.media import MediaRelay

# Streaming configurations, inspired by /opt/awscam/awsmedia/config.json
# on AWS DeepLens, which is used for AWS DeepLens' video streaming server
video_release_timeout = 0.1
live_stream_src = '/opt/awscam/out/ch1_out.h264'
max_buffer_size = 5
proj_stream_src = '/tmp/results.mjpeg'
stream_timeout = 1

VIDEO_TIME_BASE = fractions.Fraction(1, 120000)
MXUVC_BIN = "/opt/awscam/camera/installed/bin/mxuvc"



class VideoWorker(Thread):
    """ Worker thread used to read frames from the AWS DeepLens hardware.
        Inspired by /opt/awscam/awsmedia/video_server.py on AWS DeepLens.
        This class should have only one frame for a single session, then
        it should be re-initialized"""
    def __init__(self):
        super().__init__()
        self.frame_queue = queue.Queue(max_buffer_size)
        self.stop_request = Event()

    def run(self):
        while not stat.S_ISFIFO(os.stat(live_stream_src).st_mode):
            continue
        video_capture = cv2.VideoCapture(live_stream_src)
        while not self.stop_request.isSet():
            ret, frame = video_capture.read()
            try:
                if ret:
                    # Resize the frame to 480p to increase performance
                    # since when using raw 1080p frames, framerate will be low
                    frame = cv2.resize(frame, (858, 480))
                    self.frame_queue.put_nowait(frame)
            except queue.Full:
                continue
        video_capture.release()

    def get_frame(self):
        try:
            return self.frame_queue.get(timeout=stream_timeout)
        except queue.Empty:
            return None

    def join(self, timeout=None):
        self.stop_request.set()
        super().join(video_release_timeout)


class DeepLensVideoTrack(MediaStreamTrack):
    """"""
    kind = "video"

    def __init__(self, video_worker: VideoWorker):
        super().__init__()
        self._start = None
        self._worker = video_worker

    async def recv(self):
        if self.readyState != "live":
            raise MediaStreamError

        if not self._worker.is_alive():
            self._worker.start()

        current = time.time()
        if self._start is None:
            self._start = current

        diff = current - self._start
        frame_nd = self._worker.get_frame()
        if frame_nd is None:
            # No frame is available, return a green image.
            frame = VideoFrame(width=858, height=480)
        else:
            frame = VideoFrame.from_ndarray(frame_nd, format="bgr24")

        # Add pts to the frame with the self-defined time base
        # Reference: https://github.com/bonprosoft/x2webrtc/blob/master/client/x2webrtc/track.py#L75
        frame.pts = int(diff / VIDEO_TIME_BASE)
        frame.time_base = VIDEO_TIME_BASE

        return frame
