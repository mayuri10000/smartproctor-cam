#*****************************************************
#                                                    *
# Copyright 2018 Amazon.com, Inc. or its affiliates. *
# All Rights Reserved.                               *
#                                                    *
#*****************************************************
""" A sample lambda for object detection"""
import math
from threading import Thread, Event
import os
import json
import numpy as np
import awscam
import cv2

from gluoncv.data import Kinetics400Attr
import mxnet as mx
from gluoncv import model_zoo, data, utils
import time
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord


class LocalDisplay(Thread):
    """ Class for facilitating the local display of inference results
        (as images). The class is designed to run on its own thread. In
        particular the class dumps the inference results into a FIFO
        located in the tmp directory (which lambda has access to). The
        results can be rendered using mplayer by typing:
        mplayer -demuxer lavf -lavfdopts format=mjpeg:probesize=32 /tmp/results.mjpeg
    """
    def __init__(self, resolution):
        """ resolution - Desired resolution of the project stream """
        # Initialize the base class, so that the object can run on its own
        # thread.
        super(LocalDisplay, self).__init__()
        # List of valid resolutions
        RESOLUTION = {'1080p' : (1920, 1080), '720p' : (1280, 720), '480p' : (858, 480)}
        if resolution not in RESOLUTION:
            raise Exception("Invalid resolution")
        self.resolution = RESOLUTION[resolution]
        # Initialize the default image to be a white canvas. Clients
        # will update the image when ready.
        self.frame = cv2.imencode('.jpg', 255*np.ones([640, 480, 3]))[1]
        self.stop_request = Event()

    def run(self):
        """ Overridden method that continually dumps images to the desired
            FIFO file.
        """
        # Path to the FIFO file. The lambda only has permissions to the tmp
        # directory. Pointing to a FIFO file in another directory
        # will cause the lambda to crash.
        result_path = '/tmp/results.mjpeg'
        # Create the FIFO file if it doesn't exist.
        if not os.path.exists(result_path):
            os.mkfifo(result_path)
        # This call will block until a consumer is available
        with open(result_path, 'wb') as fifo_file:
            while not self.stop_request.isSet():
                try:
                    # Write the data to the FIFO file. This call will block
                    # meaning the code will come to a halt here until a consumer
                    # is available.
                    fifo_file.write(self.frame.tobytes())
                except IOError:
                    continue

    def set_frame_data(self, frame):
        """ Method updates the image data. This currently encodes the
            numpy array to jpg but can be modified to support other encodings.
            frame - Numpy array containing the image data of the next frame
                    in the project stream.
        """
        ret, jpeg = cv2.imencode('.jpg', cv2.resize(frame, self.resolution))
        if not ret:
            raise Exception('Failed to set frame data')
        self.frame = jpeg

    def join(self):
        self.stop_request.set()


label_text_color = (255, 255, 255)
label_background_color = (125, 175, 75)
box_color = (255, 128, 0)
box_thickness = 1

m_input_size = 416

yolo_scale_13 = 13
yolo_scale_26 = 26
yolo_scale_52 = 52

classes = 80
coords = 4
num = 3
anchors = [10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326]


class DetectionObject:
    xmin = 0
    ymin = 0
    xmax = 0
    ymax = 0
    class_id = 0
    confidence = 0.0

    def __init__(self, x, y, h, w, class_id, confidence, h_scale, w_scale):
        self.xmin = int((x - w / 2) * w_scale)
        self.ymin = int((y - h / 2) * h_scale)
        self.xmax = int(self.xmin + w * w_scale)
        self.ymax = int(self.ymin + h * h_scale)
        self.class_id = class_id
        self.confidence = confidence


def intersection_over_union(box_1, box_2):
    width_of_overlap_area = min(box_1.xmax, box_2.xmax) - max(box_1.xmin, box_2.xmin)
    height_of_overlap_area = min(box_1.ymax, box_2.ymax) - max(box_1.ymin, box_2.ymin)
    area_of_overlap = 0.0
    if width_of_overlap_area < 0.0 or height_of_overlap_area < 0.0:
        area_of_overlap = 0.0
    else:
        area_of_overlap = width_of_overlap_area * height_of_overlap_area
    box_1_area = (box_1.ymax - box_1.ymin)  * (box_1.xmax - box_1.xmin)
    box_2_area = (box_2.ymax - box_2.ymin)  * (box_2.xmax - box_2.xmin)
    area_of_union = box_1_area + box_2_area - area_of_overlap
    retval = 0.0
    if area_of_union <= 0.0:
        retval = 0.0
    else:
        retval = (area_of_overlap / area_of_union)
    return retval


def entry_index(side, lcoords, lclasses, location, entry):
    n = int(location / (side * side))
    loc = location % (side * side)
    return int(n * side * side * (lcoords + lclasses + 1) + entry * side * side + loc)


def parse_yolov3_result(array, resized_h, resized_w, original_h, original_w, threshold, objects):
    anchor_offset = 0
    side = 0
    if len(array) == 1 * 255 * yolo_scale_13 * yolo_scale_13:
        anchor_offset = 2 * 6
        side = yolo_scale_13
    elif len(array) == 1 * 255 * yolo_scale_26 * yolo_scale_26:
        anchor_offset = 2 * 3
        side = yolo_scale_26
    elif len(array) == 1 * 255 * yolo_scale_52 * yolo_scale_52:
        anchor_offset = 2 * 0
        side = yolo_scale_52

    side_square = side * side
    for i in range(side_square):
        row = int(i / side)
        col = int(i % side)
        for n in range(num):
            obj_index = entry_index(side, coords, classes, n * side * side + i, coords)
            box_index = entry_index(side, coords, classes, n * side * side + i, 0)
            scale = array[obj_index]
            if scale < threshold:
                continue
            x = (col + array[box_index + 0 * side_square]) / side * resized_w
            y = (row + array[box_index + 1 * side_square]) / side * resized_h
            height = math.exp(array[box_index + 3 * side_square]) * anchors[anchor_offset + 2 * n + 1]
            width = math.exp(array[box_index + 2 * side_square]) * anchors[anchor_offset + 2 * n]
            for j in range(classes):
                class_index = entry_index(side, coords, classes, n * side_square + i, coords + 1 + j)
                prob = scale * array[class_index]
                if prob < threshold:
                    continue
                obj = DetectionObject(x, y, height, width, j, prob, (original_h / resized_h),
                                      (original_w / resized_w))
                objects.append(obj)
        return objects


def infinite_infer_run():
    """ Entry point of the lambda function"""
    try:
        # This object detection model is implemented as single shot detector (ssd), since
        # the number of labels is small we create a dictionary that will help us convert
        # the machine labels to human readable labels.
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
        # Create an IoT client for sending to messages to the cloud.
        # client = greengrasssdk.client('iot-data')
        # iot_topic = '$aws/things/{}/infer'.format(os.environ['AWS_IOT_THING_NAME'])
        # Create a local display instance that will dump the image bytes to a FIFO
        # file that the image can be rendered locally.
        local_display = LocalDisplay('480p')
        local_display.start()
        # The sample projects come with optimized artifacts, hence only the artifact
        # path is required.
        model_path = '/opt/awscam/artifacts/frozen_yolo_v3.xml'
        # Load the model onto the GPU.
        # client.publish(topic=iot_topic, payload='Loading object detection model')
        print('Loading object detection model')
        model = awscam.Model(model_path, {'GPU': 1})
        # model = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)
        # model.reset_class(["person"], reuse_weights=['person'])
        # client.publish(topic=iot_topic, payload='Object detection model loaded')
        print('Object detection model loaded')
        # Set the threshold for detection
        detection_threshold = 0.15
        # The height and width of the training set images
        input_height = m_input_size
        input_width = m_input_size
        # Do inference until the lambda is killed.
        camera_width = 858
        camera_height = 480
        new_w = int(camera_width * m_input_size / camera_width)
        new_h = int(camera_height * m_input_size / camera_height)
        while True:
            # Get a frame from the video stream
            ret, frame = awscam.getLastFrame()
            if not ret:
                raise Exception('Failed to get frame from the stream')
            # Resize frame to the same size as the training set.
            frame_resize = cv2.resize(frame, (input_height, input_width))
            # Run the images through the inference engine and parse the results using
            # the parser API, note it is possible to get the output of doInference
            # and do the parsing manually, but since it is a ssd model,
            # a simple API is provided.
            result = model.doInference(frame_resize)
            # parsed_inference_results = model.parseResult(model_type,
            #                                              model.doInference(frame_resize))

            objects = []
            for name in result.keys():
                objects = parse_yolov3_result(result[name], new_h, new_w, camera_height, camera_width, 0.7, objects)

            # Dictionary to be filled with labels and probabilities for MQTT
            cloud_output = {}

            # Filtering overlapping boxes
            objlen = len(objects)
            for i in range(objlen):
                if objects[i].confidence == 0.0:
                    continue
                for j in range(i + 1, objlen):
                    if intersection_over_union(objects[i], objects[j]) >= 0.4:
                        objects[j].confidence = 0

                    # Drawing boxes
                for obj in objects:
                    if obj.confidence < 0.2:
                        continue
                    label = obj.class_id
                    confidence = obj.confidence
                    if confidence > 0.2:
                        label_text = CLASSES[label] + " (" + "{:.1f}".format(confidence * 100) + "%)"
                        cv2.rectangle(frame, (obj.xmin, obj.ymin), (obj.xmax, obj.ymax), box_color, box_thickness)
                        cv2.putText(frame, label_text, (obj.xmin, obj.ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    label_text_color, 1)

            # Set the next frame in the local display stream.
            local_display.set_frame_data(frame)
            # Send results to the cloud
            # client.publish(topic=iot_topic, payload=json.dumps(cloud_output))
            print(cloud_output)
    except Exception as ex:
        # client.publish(topic=iot_topic, payload='Error in object detection lambda: {}'.format(ex))
        print('Error in object detection lambda: {}'.format(ex))


if __name__ == '__main__':
    infinite_infer_run()
