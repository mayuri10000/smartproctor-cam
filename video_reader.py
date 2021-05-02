import os
import stat

import numpy as np
from threading import Thread, Event
import queue
import awscam
import cv2

# Streaming configurations, inspired by /opt/awscam/awsmedia/config.json
# on AWS DeepLens, which is used for AWS DeepLens' video streaming server
video_release_timeout = 0.1
live_stream_src = '/opt/awscam/out/ch1_out.h264'
max_buffer_size = 2
proj_stream_src = '/tmp/results.mjpeg'
stream_timeout = 1
stream_framerate = 15
original_framerate = 24
stream_resolution = (858, 480)
original_resolution = (1920, 1080)

MXUVC_BIN = "/opt/awscam/camera/installed/bin/mxuvc"


def set_camera_prop(fps, resolution):
    """ Helper method that sets the cameras frame rate and resolution. Used
        predominantly by the h264 video stream, should not be called if user
        is using KVS.
        fps - Desired framerate
        resolution - Tuple of (width, height) for desired resolution, accepted
                     values in RESOLUTION.
    """
    os.system("{} --ch 1 framerate {}".format(MXUVC_BIN, fps))
    os.system("{} --ch 1 resolution {} {}".format(MXUVC_BIN, resolution[0], resolution[1]))


class VideoWorker(Thread):
    """ Worker thread used to read frames from the AWS DeepLens hardware.
        Inspired by /opt/awscam/awsmedia/video_server.py on AWS DeepLens.
    """
    def __init__(self):
        super().__init__()
        self.frame_queue = queue.Queue(max_buffer_size)
        self.stop_request = Event()
        self.tracks = set()

    def run(self):
        set_camera_prop(stream_framerate, stream_resolution)
        while not stat.S_ISFIFO(os.stat(live_stream_src).st_mode):
            continue
        video_capture = cv2.VideoCapture(live_stream_src)
        while not self.stop_request.isSet():
            ret, frame = video_capture.read()
            try:
                if ret:
                    jpeg = cv2.imencode('.jpg', frame)[1]
                    self.frame_queue.put_nowait(jpeg)
            except queue.Full:
                continue
        video_capture.release()

    def get_frame(self):
        """ Gets one JPEG video frame, a pure-black frame if the queue is empty """
        try:
            return self.frame_queue.get(timeout=stream_timeout).tobytes()
        except queue.Empty:
            black_canvas = 0 * np.ones([stream_resolution[0], stream_resolution[1], 3])
            jpeg = cv2.imencode('.jpg', cv2.resize(black_canvas, stream_resolution))[1]
            return jpeg.tobytes()

    def join(self, timeout=None):
        self.stop_request.set()
        set_camera_prop(original_framerate, original_resolution)
        super().join(video_release_timeout)
