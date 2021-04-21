import threading
from typing import Optional, Set

import av
from av import AudioFrame, VideoFrame
import asyncio
import time
import awscam
import cv2
from aiortc.mediastreams import MediaStreamTrack, MediaStreamError, VideoStreamTrack

# This is the FIFO pipe that outputs the raw frames of the camera capture
DEEPLENS_VIDEO_FIFO_NAME = "/opt/awscam/out/ch1_out.h264"


class DeepLensVideoTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.capture = cv2.VideoCapture(DEEPLENS_VIDEO_FIFO_NAME)

    async def recv(self):
        if self.readyState != "live":
            raise MediaStreamError

        ret, cv_frame = self.capture.read()
        # ret, cv_frame = awscam.getLastFrame()
        if not ret:
            self.stop()
            raise MediaStreamError
        frame = VideoFrame.from_ndarray(cv_frame, format="bgr24")

        pts, time_base = await self.next_timestamp()
        frame.pts = pts
        frame.time_base = time_base

        return frame