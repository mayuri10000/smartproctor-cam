# *****************************************************
#                                                    *
# Copyright 2018 Amazon.com, Inc. or its affiliates. *
# All Rights Reserved.                               *
#                                                    *
# *****************************************************
""" A sample lambda for object detection"""
import math
from threading import Thread, Event
import os
import json
import numpy as np
import awscam
import cv2
import time

import mxnet as mx
from gluoncv import model_zoo, data, utils
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord


CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
           'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
           'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
           'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
           'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
           'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
           'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
           'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
           'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
           'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
           'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
           'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
           'scissors', 'teddy bear', 'hair drier', 'toothbrush']


__message_callback = lambda msg: print(msg)
__should_stop = False


def set_message_callback(callback):
    global __message_callback
    __message_callback = callback


def stop_inference():
    global __should_stop
    __should_stop = True


def infinite_infer_run():
    global __should_stop
    # currently the model is slow since it is not optimized.
    # so these events will be triggered if detected in one frame.
    # But we should detect in more than one frame since error could occur in certain frames
    multiple_monitors_num = 0
    multiple_monitors_max = 2
    no_person_num = 0
    no_person_max = 2
    multiple_person_num = 0
    multiple_person_max = 2
    cellphones_num = 0
    cellphones_max = 1
    book_num = 0
    book_max = 1

    __should_stop = False
    """ Entry point of the lambda function"""
    try:
        detector = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)
        print("Model loaded, start inferencing")
        while not __should_stop:
            ret, frame = awscam.getLastFrame()
            if not ret:
                raise Exception("Failed to get frame from the stream")
            img = mx.nd.array(frame)
            x, img = data.transforms.presets.ssd.transform_test(img, short=256)
            class_IDs, scores, bounding_boxs = detector(x)
            persons = []
            cellphones = []
            monitors = []
            books = []

            mx.nd.waitall()
            for x in range(len(class_IDs[0])):
                if scores[0][x] > 0.2:
                    class_name = CLASSES[int(class_IDs[0][x].asscalar())]
                    if class_name == 'person':
                        persons.append({
                            'bounding_box': bounding_boxs[0][x]
                        })
                    elif class_name == 'tv' or class_name == 'laptop':
                        monitors.append({
                            'bounding_box': bounding_boxs[0][x]
                        })
                    elif class_name == 'cell phone':
                        cellphones.append({
                            'bounding_box': bounding_boxs[0][x]
                        })
                    elif class_name == 'book':
                        books.append({
                            'bounding_box': bounding_boxs[0][x]
                        })

            if len(persons) == 0:
                no_person_num += 1
            else:
                no_person_num = 0

            if len(persons) > 1:
                multiple_person_num += 1
            else:
                multiple_person_num = 0

            if len(cellphones) > 0:
                cellphones_num += 1
            else:
                cellphones_num = 0

            if len(books) > 0:
                book_num += 1
            else:
                book_num = 0

            if len(monitors) > 1:
                multiple_monitors_num += 1
            else:
                multiple_monitors_num = 0

            if multiple_person_num == multiple_person_max:
                __message_callback('multiple person detected')

            if no_person_num == no_person_max:
                __message_callback('exam taker left')

            if multiple_monitors_num == multiple_monitors_max:
                __message_callback('multiple monitors detected')

            if book_num == book_max:
                __message_callback('books detected')

            if cellphones_num == cellphones_max:
                __message_callback('cellphones detected')
        print('inference stop')
    except Exception as ex:
        # client.publish(topic=iot_topic, payload='Error in object detection lambda: {}'.format(ex))
        print('Error in object detection lambda: {}'.format(ex))


if __name__ == '__main__':
    infinite_infer_run()
