import math
from threading import Thread, Event
import os
import json
import numpy as np
import awscam
import cv2
import time

from video_reader import VideoWorker, _DeepLensVideoTrack


# The model used in the project is pre-trained with the COCO dataset
# The labels in the COCO dataset can be found in
# https://github.com/ActiveState/gococo/blob/master/labels.txt
PERSON_LABEL = 1
TV_LABEL = 72
LAPTOP_LABEL = 73
CELLPHONE_LABEL = 77
BOOK_LABEL = 84

# We have different threshold for different objects since the
# model's accuracy varies with the object detected
PERSON_THRESHOLD = 0.5
TV_LAPTOP_THRESHOLD = 0.5
CELLPHONE_THRESHOLD = 0.1
BOOK_THRESHOLD = 0.4

# Errors could occur during detection, but normally they will not occur in many continuous frames
# We count the occurrences of different situations and detect whether the count exceeds a threshold
# Also, the number of frames where the previously detected situation discontinues, when the count exceeds
# a limit, we regard the previous situation as ended
no_person_max = 10
no_person_discontinue_max = 10
multi_person_max = 20
multi_person_discontinue_max = 20
multi_monitor_max = 10
multi_monitor_discontinue_max = 10
cellphone_max = 3
cellphone_discontinue_max = 10
book_max = 3
book_discontinue_max = 10

# The path to the optimized model, should be in /opt/awscam/artifacts/ when deployed
model_path = '/home/aws_cam/public/ssd_mobilenet_v2_coco/FP16/ssd_mobilenet_v2_coco.xml'
model_type = 'ssd'
input_height = 300
input_width = 300


class InferenceWorker(Thread):
    """ Worker thread that do the object detection inference."""
    def __init__(self, video_track: _DeepLensVideoTrack, message_callback=print):
        super().__init__()
        self.no_person_count = 0
        self.no_person_discontinue = 0
        self.multi_person_count = 0
        self.multi_person_discontinue = 0
        self.multi_monitor_count = 0
        self.multi_monitor_discontinue = 0
        self.cellphone_count = 0
        self.cellphone_discontinue = 0
        self.book_count = 0
        self.book_discontinue = 0
        self.model = None
        self.video_track = video_track
        self.stop_request = Event()
        self.message_callback = message_callback
        self.yscale = 0
        self.xscale = 0

    def run(self):
        # Load the optimized object detection model
        self.model = awscam.Model(model_path, {'GPU': 1})
        while not self.stop_request.isSet():
            # We do not use awscam.getLastFrame since having multiple consumer of the video FIFO
            # will make the video output corrupt. Should share the same VideoWorker instance
            # with the WebRTC media track
            frame = self.video_track.recv_inference()
            if frame is None:
                continue

            frame_resize = cv2.resize(frame, (input_height, input_width))
            # Process the frame data with the object detection model and parse the result with
            # the AWS DeepLens' builtin API.
            result = self.model.parseResult(model_type, self.model.doInference(frame_resize))
            self.yscale = float(frame.shape[0]) / float(input_height)
            self.xscale = float(frame.shape[1]) / float(input_width)
            self.process_result(result)

    def process_result(self, result):
        persons = []
        monitors = []
        cellphones = []
        books = []
        # Get the detected objects and probabilities
        for obj in result[model_type]:
            # Add bounding boxes to full resolution frame
            xmin = int(self.xscale * obj['xmin'])
            ymin = int(self.yscale * obj['ymin'])
            xmax = int(self.xscale * obj['xmax'])
            ymax = int(self.yscale * obj['ymax'])

            if obj['label'] == PERSON_LABEL and obj['prob'] > PERSON_THRESHOLD:
                persons.append((xmin, xmax, ymin, ymax, obj['prob']))
            elif (obj['label'] == TV_LABEL or obj['label'] == LAPTOP_LABEL) and obj['prob'] > TV_LAPTOP_THRESHOLD:
                monitors.append((xmin, xmax, ymin, ymax, obj['prob']))
            elif obj['label'] == CELLPHONE_LABEL and obj['prob'] > CELLPHONE_THRESHOLD:
                cellphones.append((xmin, xmax, ymin, ymax, obj['prob']))
            elif obj['label'] == BOOK_LABEL and obj['prob'] > BOOK_THRESHOLD:
                books.append((xmin, xmax, ymin, ymax, obj['prob']))

        if len(persons) < 1:
            self.no_person_count += 1
            self.no_person_discontinue = 0
        elif self.no_person_count > 0:
            self.no_person_discontinue += 1

        if self.no_person_discontinue == no_person_discontinue_max:
            self.no_person_count = 0
            self.no_person_discontinue = 0

        if len(persons) > 1:
            self.multi_person_count += 1
            self.multi_person_discontinue = 0
        elif self.multi_person_count > 0:
            self.multi_person_discontinue += 1

        if self.multi_person_discontinue == multi_person_discontinue_max:
            self.multi_person_count = 0
            self.multi_person_discontinue = 0

        if len(monitors) > 1:
            self.multi_monitor_count += 1
            self.multi_monitor_discontinue = 0
        elif self.multi_monitor_count > 0:
            self.multi_monitor_discontinue += 1

        if self.multi_monitor_discontinue == multi_monitor_discontinue_max:
            self.multi_monitor_count = 0
            self.multi_monitor_discontinue = 0

        if len(cellphones) > 0:
            self.cellphone_count += 1
            self.cellphone_discontinue = 0
        elif self.cellphone_count > 0:
            self.cellphone_discontinue += 1

        if self.cellphone_discontinue == cellphone_discontinue_max:
            self.cellphone_count = 0
            self.cellphone_discontinue = 0

        if len(books) > 1:
            self.book_count += 1
            self.book_discontinue = 0
        elif self.book_count > 0:
            self.book_discontinue += 1

        if self.book_discontinue == book_discontinue_max:
            self.book_count = 0
            self.book_discontinue = 0

        if self.no_person_count == no_person_max:
            self.message_callback('Exam taker left')
            # save_image(frame, persons, 'Exam taker left')
            self.no_person_count += 1

        if self.multi_person_count == multi_person_max:
            self.message_callback('multiple people detected')
            # save_image(frame, persons, 'multiple people detected')
            self.multi_person_count += 1

        if self.multi_monitor_count == multi_monitor_max:
            self.message_callback('multiple PC monitors/laptops detected')
            # save_image(frame, monitors, 'multiple PC monitors/laptops detected')
            self.multi_monitor_count += 1

        if self.cellphone_count == cellphone_max:
            self.message_callback('cellphone detected')
            # save_image(frame, cellphones, 'cellphone detected')
            self.cellphone_count += 1

        if self.book_count == book_max:
            self.message_callback('book detected')
            # save_image(frame, books, 'book detected')
            self.book_count += 1

    def join(self, timeout=None):
        self.stop_request.set()
        super().join(timeout)


if __name__ == '__main__':
    video_worker = VideoWorker()
    worker = InferenceWorker(video_worker.get_track())
    worker.start()
    input()
