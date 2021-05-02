import os

import requests
from flask import Flask, Response, render_template, jsonify, request

import utils
from video_reader import VideoWorker
from inference import InferenceWorker


# Server address, should be changed to DNS name if deployed
SERVER_ADDR = "10.28.140.146:5001"
SERVER_PROTOCOL = 'https'


class SmartProctorApp:
    """ The interface of the SmartProctor edge computing client, enabling the web client to
     interact with the edge computing client """
    def __init__(self):
        self.exam_id = 0
        self.video_worker = None
        self.inference_worker = None
        self.app = Flask("smartproctor-cam")
        self.app.add_url_rule('/sn', 'sn', self.get_serial, methods=['GET'])
        self.app.add_url_rule("/login_and_start_exam", 'login_and_start_exam', self.login_and_start_exam, methods=['POST'])
        self.app.add_url_rule('/stop_exam', 'stop_exam', self.stop_exam, methods=['GET'])
        self.app.add_url_rule('/network_status', 'network_status', self.network_status, methods=['GET'])
        self.app.add_url_rule('/connect_wifi', 'connect_wifi', self.connect_wifi, methods=['POST'])
        self.app.add_url_rule('/wifi_ssids', 'wifi_ssids', self.ssids, methods=['GET'])
        self.app.add_url_rule('/video_stream', 'video_stream', self.video_stream, methods=['GET'])
        self.app.after_request(self.add_cors_header)

    def run_server(self, port=8080):
        self.app.run(host='0.0.0.0', port=port, threaded=True)

    def add_cors_header(self, response):
        """ Adds CORS headers to the response, if no CORS header present, the server
         cannot be accessed with the web client """
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-CSRF-Token, Cache-Control'
        return response

    def get_serial(self):
        """ Gets the device's serial number """
        return jsonify({"serialNumber": utils.get_device_serial_number()})

    def ssids(self):
        """ Gets a list of wifi SSIDs """
        return jsonify({'wifiList': utils.list_ssid()})

    def connect_wifi(self):
        """ Connect to a specific WIFI hotspot with SSID and password """
        params = request.get_json()
        return jsonify({'success': utils.connect_wifi(params['ssid'], params['password'])})

    def network_status(self):
        """ Get current network status, with IP address """
        return jsonify(utils.get_network_status())

    def login_and_start_exam(self):
        """ Logs in to the SmartProctor's server and begin the exam """
        try:
            params = request.get_json()
            # Log in to the server with the token from web client
            res = requests.get(f"{SERVER_PROTOCOL}://{SERVER_ADDR}/api/user/DeepLensLogin/" + params['token'], verify=False)
            o = res.json()
            if o['code'] == 0:
                self.exam_id = params['examId']
                # Stop the inference worker thread if running
                if self.inference_worker is not None and self.inference_worker.is_alive():
                    self.inference_worker.join()

                # Start the video worker thread if not started
                if self.video_worker is None or not self.video_worker.is_alive():
                    self.video_worker = VideoWorker()
                    self.video_worker.start()

                # Obtains the auth cookie from the login response
                cookie = res.headers['Set-Cookie']
                exam_details = requests.get(f"{SERVER_PROTOCOL}://{SERVER_ADDR}/api/exam/ExamDetails/" + str(params['examId']),
                                            verify=False).json()

                # Begin inference
                self.inference_worker = InferenceWorker(self.exam_id, exam_details['openBook'], cookie)
                self.inference_worker.start()
            return jsonify({"success": o['code'] == 0})
        except Exception as e:
            print(str(e))
            return jsonify({"success": False})

    def stop_exam(self):
        """ Stop the exam, stop the worker therads """
        if self.video_worker is not None and self.video_worker.is_alive():
            self.video_worker.join()
        if self.inference_worker is not None and self.inference_worker.is_alive():
            self.inference_worker.join()

        self.video_worker = None
        self.inference_worker = None

    def __gen_video_stream(self):
        while True:
            # Gets the camera frames from the video worker
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n'
                   + self.video_worker.get_frame() + b'\r\n')

    def video_stream(self):
        """ Get the MJPEG video stream """
        # Start the video worker thread if not started
        if self.video_worker is None or not self.video_worker.is_alive():
            self.video_worker = VideoWorker()
            self.video_worker.start()

        return Response(self.__gen_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


# This script should be run as root since it requires access to "iptables" and
# "mxuvc". So this project cannot be deployed to the DeepLens with AWS console.
# Instead, it should be manually installed.
if __name__ == '__main__':
    # Allows the port used by the server
    os.system("iptables -A INPUT -p tcp --dport 8080 -j ACCEPT")
    app = SmartProctorApp()
    app.run_server()
