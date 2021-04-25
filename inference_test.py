""" A sample lambda for object detection"""
from threading import Thread, Event
import os
import json
import numpy as np
import awscam
import cv2
import time
from datetime import datetime
from video_reader import VideoWorker


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
        RESOLUTION = {'1080p': (1920, 1080), '720p': (1280, 720), '480p': (858, 480)}
        if resolution not in RESOLUTION:
            raise Exception("Invalid resolution")
        self.resolution = RESOLUTION[resolution]
        # Initialize the default image to be a white canvas. Clients
        # will update the image when ready.
        self.frame = cv2.imencode('.jpg', 255 * np.ones([640, 480, 3]))[1]
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
        with open(result_path, 'w') as fifo_file:
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
no_person_num = 0
no_person_max = 30
no_person_discontinue = 0
no_person_discontinue_max = 30
multi_person_num = 0
multi_person_max = 20
multi_person_discontinue = 0
multi_person_discontinue_max = 20
multi_monitor_num = 0
multi_monitor_max = 10
multi_monitor_discontinue = 0
multi_monitor_discontinue_max = 10
cellphone_num = 0
cellphone_max = 3
cellphone_discontinue = 0
cellphone_discontinue_max = 30
book_num = 0
book_max = 3
book_discontinue = 0
book_discontinue_max = 30


def save_image(frame, boxes, text):
    cv2.putText(frame, text, (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 6)
    cv2.putText(frame, str(datetime.now()), (0, 130), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 6)
    for box in boxes:
        xmin, xmax, ymin, ymax, score = box
        # See https://docs.opencv.org/3.4.1/d6/d6e/group__imgproc__draw.html
        # for more information about the cv2.rectangle method.
        # Method signature: image, point1, point2, color, and tickness.
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 10)
        # Amount to offset the label/probability text above the bounding box.
        text_offset = 15
        # See https://docs.opencv.org/3.4.1/d6/d6e/group__imgproc__draw.html
        # for more information about the cv2.putText method.
        # Method signature: image, text, origin, font face, font scale, color,
        # and tickness
        cv2.putText(frame, "{:.2f}%".format(score * 100),
                    (xmin, ymin - text_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 6)
    cv2.imwrite('/home/aws_cam/detection/' + str(int(time.time())) + ".jpg", frame)


def infinite_infer_run():
    global no_person_num, no_person_discontinue, multi_person_num, \
        multi_person_discontinue, cellphone_num, cellphone_discontinue, \
        book_num, book_discontinue, multi_monitor_num, multi_monitor_discontinue
    """ Entry point of the lambda function"""
    try:
        # This object detection model is implemented as single shot detector (ssd), since
        # the number of labels is small we create a dictionary that will help us convert
        # the machine labels to human readable labels.
        model_type = 'ssd'

        # Create a local display instance that will dump the image bytes to a FIFO
        # file that the image can be rendered locally.
        local_display = LocalDisplay('480p')
        local_display.start()

        video_worker = VideoWorker()
        track = video_worker.get_track(buffer_size=1)
        # The sample projects come with optimized artifacts, hence only the artifact
        # path is required.
        model_path = '/home/aws_cam/public/ssd_mobilenet_v2_coco/FP16/ssd_mobilenet_v2_coco.xml'
        # Load the model onto the GPU.
        print('Loading object detection model')
        start = time.time()
        model = awscam.Model(model_path, {'GPU': 1})
        print('Object detection model loaded, Elapsed: ' + str(time.time() - start))
        # The height and width of the training set images
        input_height = 300
        input_width = 300
        # Do inference until the lambda is killed.
        while True:
            # Get a frame from the video stream
            frame = track.recv_inference()
            if frame is None:
                continue
            # Resize frame to the same size as the training set.
            start = time.time()
            frame_resize = cv2.resize(frame, (input_height, input_width))
            # Run the images through the inference engine and parse the results using
            # the parser API, note it is possible to get the output of doInference
            # and do the parsing manually, but since it is a ssd model,
            # a simple API is provided.
            parsed_inference_results = model.parseResult(model_type,
                                                         model.doInference(frame_resize))
            elapsed = time.time() - start
            # Compute the scale in order to draw bounding boxes on the full resolution
            # image.
            yscale = float(frame.shape[0]) / float(input_height)
            xscale = float(frame.shape[1]) / float(input_width)

            persons = []
            monitors = []
            cellphones = []
            books = []
            # Get the detected objects and probabilities
            for obj in parsed_inference_results[model_type]:
                # Add bounding boxes to full resolution frame
                xmin = int(xscale * obj['xmin'])
                ymin = int(yscale * obj['ymin'])
                xmax = int(xscale * obj['xmax'])
                ymax = int(yscale * obj['ymax'])

                if obj['label'] == PERSON_LABEL and obj['prob'] > PERSON_THRESHOLD:
                    persons.append((xmin, xmax, ymin, ymax, obj['prob']))
                elif (obj['label'] == TV_LABEL or obj['label'] == LAPTOP_LABEL) and obj['prob'] > TV_LAPTOP_THRESHOLD:
                    monitors.append((xmin, xmax, ymin, ymax, obj['prob']))
                elif obj['label'] == CELLPHONE_LABEL and obj['prob'] > CELLPHONE_THRESHOLD:
                    cellphones.append((xmin, xmax, ymin, ymax, obj['prob']))
                elif obj['label'] == BOOK_LABEL and obj['prob'] > BOOK_THRESHOLD:
                    books.append((xmin, xmax, ymin, ymax, obj['prob']))

            if len(persons) < 1:
                no_person_num += 1
                no_person_discontinue = 0
            elif no_person_num > 0:
                no_person_discontinue += 1

            if no_person_discontinue == no_person_discontinue_max:
                no_person_num = 0
                no_person_discontinue = 0

            if len(persons) > 1:
                multi_person_num += 1
                multi_person_discontinue = 0
            elif multi_person_num > 0:
                multi_person_discontinue += 1

            if multi_person_discontinue == multi_person_discontinue_max:
                multi_person_num = 0
                multi_person_discontinue = 0

            if len(monitors) > 1:
                multi_monitor_num += 1
                multi_monitor_discontinue = 0
            elif multi_monitor_num > 0:
                multi_monitor_discontinue += 1

            if multi_monitor_discontinue == multi_monitor_discontinue_max:
                multi_monitor_num = 0
                multi_monitor_discontinue = 0

            if len(cellphones) > 0:
                cellphone_num += 1
                cellphone_discontinue = 0
            elif cellphone_num > 0:
                cellphone_discontinue += 1

            if cellphone_discontinue == cellphone_discontinue_max:
                cellphone_num = 0
                cellphone_discontinue = 0

            if len(books) > 1:
                book_num += 1
                book_discontinue = 0
            elif book_num > 0:
                book_discontinue += 1

            if book_discontinue == book_discontinue_max:
                book_num = 0
                book_discontinue = 0

            if no_person_num == no_person_max:
                print('Exam taker left')
                save_image(frame, persons, 'Exam taker left')
                no_person_num += 1

            if multi_person_num == multi_person_max:
                print('multiple people detected')
                save_image(frame, persons, 'multiple people detected')
                multi_person_num += 1

            if multi_monitor_num == multi_monitor_max:
                print('multiple PC monitors/laptops detected')
                save_image(frame, monitors, 'multiple PC monitors/laptops detected')
                multi_monitor_num += 1

            if cellphone_num == cellphone_max:
                print('cellphone detected')
                save_image(frame, cellphones, 'cellphone detected')
                cellphone_num += 1

            if book_num == book_max:
                print('book detected')
                save_image(frame, books, 'book detected')
                book_num += 1
    except Exception as ex:
        print('Error in object detection lambda: {}'.format(ex))


if __name__ == '__main__':
    infinite_infer_run()
