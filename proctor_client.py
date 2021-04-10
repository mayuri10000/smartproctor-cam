import logging

import time
import aiortc
import aiohttp
import json
from signalrcore.hub_connection_builder import HubConnectionBuilder

from video_reader import DeepLensVideoReader

reader = DeepLensVideoReader()
proctor_connections = {}
test_taker_connection = aiortc.RTCPeerConnection()
signalr_conn = None
http_session = aiohttp.ClientSession()

server_name = '10.105.168.218:5001'

hub_url = f"wss://{server_name}/hub"

auth_cookie = None
exam_id = 0


async def login(token, eid):
    global auth_cookie, exam_id
    params = {'token': token}
    res = await http_session.get(f"https://{server_name}/api/user/DeepLensLogin/" + token, verify_ssl=False)
    o = json.loads(await res.text())
    if o['code'] != 0:
        return False

    auth_cookie = res.headers["Set-Cookie"]
    exam_id = eid
    return True


async def init_signalr():
    global signalr_conn, proctor_connections
    signalr_conn = HubConnectionBuilder().with_url(hub_url, options=
    {'headers': {'Cookies': auth_cookie}, 'verify_ssl': False}) \
        .configure_logging(logging.DEBUG) \
        .build()

    signalr_conn.on_error = lambda error: print("Error! " + error)
    signalr_conn.on_disconnect = lambda: print("Disconnected")

    async def camera_answer_from_taker(sdp):
        await test_taker_connection.setRemoteDescription(sdp)

    async def camera_answer_from_proctor(proctor, sdp):
        await proctor_connections[proctor].setRemoteDescription(sdp)

    async def camera_ice_candidate_from_taker(candidate):
        await test_taker_connection.addIceCandidate(candidate)

    async def camera_ice_candidate_from_proctor(proctor, candidate):
        await proctor_connections[proctor].addIceCandidate(candidate)

    async def test_back(uid):
        print(uid)

    signalr_conn.on("CameraAnswerFromTaker", camera_answer_from_taker)
    signalr_conn.on("CameraAnswerFromProctor", camera_answer_from_proctor)
    signalr_conn.on("CameraIceCandidateFromTaker", camera_ice_candidate_from_taker)
    signalr_conn.on("CameraIceCandidateFromProctor", camera_ice_candidate_from_proctor)
    signalr_conn.on("TestBack", lambda uid: print(uid))

    signalr_conn.start()
    time.sleep(5)
    signalr_conn.send("Test", [])


async def init_webrtc(proctors):
    test_taker_connection.addTrack(reader.video)
    taker_sdp = await test_taker_connection.createOffer()
    # signalr_conn.send("CameraOfferToTaker", [taker_sdp])
    await test_taker_connection.setLocalDescription(taker_sdp)

    for proctor in proctors:
        conn = aiortc.RTCPeerConnection()
        conn.addTrack(reader.video)
        proctor_connections[proctor] = conn
        sdp = await conn.createOffer()
        # signalr_conn.send("CameraOfferToProctor", [proctor, sdp])
        await conn.setLocalDescription(sdp)


async def init_exam():
    res = await http_session.get(f"https://{server_name}/api/exam/GetProctors/" + str(3), verify_ssl=False)
    o = json.loads(await res.text())
    if o['code'] == 0:
        proctors = o['proctors']
        await init_signalr()
        await init_webrtc(proctors)


def shutdown():
    test_taker_connection.close()
    for proctor in proctor_connections.keys():
        proctor_connections[proctor].close()

    signalr_conn.stop()
