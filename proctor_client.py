import threading
import aiortc
import aiohttp
import json

from aiortc import RTCSessionDescription
from aiortc.rtcicetransport import *
from signalrcore.hub_connection_builder import HubConnectionBuilder

from video_reader import DeepLensVideoTrack, VideoWorker
from inference import InferenceWorker


proctor_connections = {}
test_taker_connection = aiortc.RTCPeerConnection()
signalr_conn = None
http_session = aiohttp.ClientSession()

server_name = '10.105.168.218:5001'

hub_url = f"wss://{server_name}/hub"

auth_cookie = None
exam_id = 0

inference_worker = None
video_worker = None


def send_detect_event(message):
    if signalr_conn:
        signalr_conn.send("TestTakerMessage", [str(exam_id), 'warning', message])


async def upload_detection_image(data):
    res = await http_session.post(f"https://{server_name}/api/exam/UploadDetection/", data, verify_ssl=False)
    o = json.loads(await res.text())
    if o['code'] != 0:
        return None
    file_name = o['fileName']
    return file_name


async def login(token, eid):
    global auth_cookie, exam_id
    res = await http_session.get(f"https://{server_name}/api/user/DeepLensLogin/" + token, verify_ssl=False)
    o = json.loads(await res.text())
    if o['code'] != 0:
        return False

    auth_cookie = res.headers["Set-Cookie"]
    exam_id = eid
    return True


def init_inference():
    global inference_worker, video_worker
    if inference_worker is not None and inference_worker.is_alive():
        inference_worker.join()
    if video_worker is not None and video_worker.is_alive():
        video_worker.join()
    video_worker = VideoWorker()
    inference_worker = InferenceWorker(video_worker, message_callback=send_detect_event)
    inference_worker.start()


async def init_signalr():
    global signalr_conn, proctor_connections
    if signalr_conn is not None:
        signalr_conn.stop()

    signalr_conn = HubConnectionBuilder().with_url(hub_url, options=
    {'headers': {'Cookie': auth_cookie}, 'verify_ssl': False}) \
        .configure_logging(logging.DEBUG)\
        .with_automatic_reconnect({
            "type": "raw",
            "keep_alive_interval": 10,
            "reconnect_interval": 5,
            "max_attempts": 5
         })\
        .build()

    def camera_answer_from_taker(args):
        sdp = RTCSessionDescription(sdp=args[0]['sdp'], type=args[0]['type'])
        print("Get SDP answer from test taker")
        asyncio.run(test_taker_connection.setRemoteDescription(sdp))

    def camera_answer_from_proctor(args):
        print("Get SDP answer from proctor " + args[0])
        sdp = RTCSessionDescription(sdp=args[1]['sdp'], type=args[1]['type'])
        asyncio.run(proctor_connections[args[0]].setRemoteDescription(sdp))

    #def camera_ice_candidate_from_taker(args):
    #    print("Get ICE candidate from test taker")
    #    loop = asyncio.get_event_loop()
    #    candidate = Candidate.from_sdp()
    #    loop.run_in_executor(None, test_taker_connection.addIceCandidate, sdp)

    def camera_ice_candidate_from_proctor(args):
        proctor_connections[args[0]].addIceCandidate(args[1])

    def exam_ended(_):
        shutdown()

    # def test_back(args):
    #     print("TestBack" + args[0])

    signalr_conn.on("CameraAnswerFromTaker", camera_answer_from_taker)
    signalr_conn.on("CameraAnswerFromProctor", camera_answer_from_proctor)
    # signalr_conn.on("CameraIceCandidateFromTaker", camera_ice_candidate_from_taker)
    # signalr_conn.on("CameraIceCandidateFromProctor", camera_ice_candidate_from_proctor)
    signalr_conn.on("ExamEnded", exam_ended)
    signalr_conn.start()


async def init_webrtc(proctors):
    global test_taker_connection, proctor_connections

    # if there is previous instances, clear them
    if test_taker_connection is not None:
        await test_taker_connection.close()
        test_taker_connection = aiortc.RTCPeerConnection()
    if len(proctor_connections) > 0:
        for key in proctor_connections:
            await proctor_connections[key].close()
        proctor_connections = {}

    test_taker_connection.addTrack(DeepLensVideoTrack(video_worker))
    taker_sdp = await test_taker_connection.createOffer()
    signalr_conn.send("CameraOfferToTaker", [taker_sdp])
    await test_taker_connection.setLocalDescription(taker_sdp)

    for proctor in proctors:
        conn = aiortc.RTCPeerConnection()
        conn.addTrack(DeepLensVideoTrack(video_worker))
        proctor_connections[proctor['id']] = conn
        sdp = await conn.createOffer()
        signalr_conn.send("CameraOfferToProctor", [proctor['id'], sdp])
        await conn.setLocalDescription(sdp)


async def init_exam():
    res = await http_session.get(f"https://{server_name}/api/exam/GetProctors/" + str(3), verify_ssl=False)
    o = json.loads(await res.text())
    if o['code'] == 0:
        proctors = o['proctors']
        init_inference()
        await init_signalr()
        await init_webrtc(proctors)


def shutdown():
    test_taker_connection.close()
    for proctor in proctor_connections.keys():
        proctor_connections[proctor].close()

    inference_worker.join()
    video_worker.join()
    signalr_conn.stop()
