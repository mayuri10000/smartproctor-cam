from threading import Thread, Event
import awscam
import cv2
import requests
import json

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
model_path = '/opt/smartpoctor/Model/ssd_mobilenet_v2_coco.xml'
model_type = 'ssd'
input_height = 300
input_width = 300

# Server address, should be changed to DNS name if deployed
SERVER_ADDR = "10.28.140.146"
SERVER_PROTOCOL = 'http'
SERVER_URL = SERVER_PROTOCOL + '://' + SERVER_ADDR


class InferenceWorker(Thread):
    """ Worker thread that do the object detection inference."""
    def __init__(self, exam_id, allow_books, auth_cookie):
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
        self.stop_request = Event()
        self.exam_id = exam_id
        self.allow_books = allow_books
        self.auth_cookie = auth_cookie
        self.yscale = 0
        self.xscale = 0

    def run(self):
        # Load the optimized object detection model
        self.model = awscam.Model(model_path, {'GPU': 1})
        while not self.stop_request.isSet():
            res, frame = awscam.getLastFrame()
            if not res:
                continue

            frame_resize = cv2.resize(frame, (input_height, input_width))
            # Process the frame data with the object detection model and parse the result with
            # the AWS DeepLens' builtin API.
            result = self.model.parseResult(model_type, self.model.doInference(frame_resize))
            self.yscale = float(frame.shape[0]) / float(input_height)
            self.xscale = float(frame.shape[1]) / float(input_width)
            self.process_result(result, frame)

    def __upload_frame(self, frame):
        try:
            files = {'file': ('detection.jpg', frame, 'image/jpeg')}
            res = requests.post(SERVER_URL + "/api/exam/UploadEventAttachment", files=files,
                                headers={'Cookie': self.auth_cookie}, verify=False)
            o = res.json()
            return o['fileName']
        except:
            return None

    def __send_event_with_frame(self, message, frame):
        file_name = self.__upload_frame(frame)
        res = requests.post(SERVER_URL + '/api/exam/SendEvent', json={
            'examId': self.exam_id,
            'type': 1,
            'receipt': None,
            'message': message,
            'attachment': file_name
        }, headers={'Cookie': self.auth_cookie}, verify=False)

    def mark_frame(self, frame, boxes, text):
        cv2.putText(frame, text, (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 6)
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
        return cv2.imencode('.jpg', frame)[1].tobytes()

    def process_result(self, result, frame):
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

        if len(books) > 1 and not self.allow_books:
            self.book_count += 1
            self.book_discontinue = 0
        elif self.book_count > 0:
            self.book_discontinue += 1

        if self.book_discontinue == book_discontinue_max:
            self.book_count = 0
            self.book_discontinue = 0

        if self.no_person_count == no_person_max:
            self.__send_event_with_frame('Exam taker left', self.mark_frame(frame, [], 'Exam taker left'))
            self.no_person_count += 1

        if self.multi_person_count == multi_person_max:
            self.__send_event_with_frame('multiple people detected', self.mark_frame(frame, persons, 'multiple people detected'))
            self.multi_person_count += 1

        if self.multi_monitor_count == multi_monitor_max:
            self.__send_event_with_frame('multiple PC monitors/laptops detected', self.mark_frame(frame, monitors, 'multiple PC monitors/laptops detected'))
            self.multi_monitor_count += 1

        if self.cellphone_count == cellphone_max:
            self.__send_event_with_frame('cellphone detected', self.mark_frame(frame, cellphones, 'cellphone detected'))
            self.cellphone_count += 1

        if self.book_count == book_max:
            self.__send_event_with_frame('book detected', self.mark_frame(frame, books, 'book detected'))
            self.book_count += 1

    def join(self, timeout=None):
        self.stop_request.set()
        super().join(timeout)
