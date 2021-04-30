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
                    jpeg = cv2.imencode('.jpg', frame)[1]
                    self.frame_queue.put_nowait(jpeg)
            except queue.Full:
                continue
        video_capture.release()

    def get_frame(self):
        try:
            return self.frame_queue.get(timeout=stream_timeout).tobytes()
        except queue.Empty:
            black_canvas = 0 * np.ones([858, 480, 3])
            jpeg = cv2.imencode('.jpg', cv2.resize(black_canvas, (858, 480)))[1]
            return jpeg.tobytes()

    def join(self, timeout=None):
        self.stop_request.set()
        super().join(video_release_timeout)
