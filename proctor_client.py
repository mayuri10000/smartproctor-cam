import aiortc
import aiohttp
import json
from signalrcore_async.hub_connection_builder import HubConnectionBuilder

from video_reader import DeepLensVideoReader

reader = DeepLensVideoReader()
proctor_connections = {}
test_taker_connection = aiortc.RTCPeerConnection()
signalr_conn = None
http_session = aiohttp.ClientSession()

server = '192.168.1.2'

hub_url = f"ws://{server}/hub"

auth_cookie = None


async def login(uid, password):
    global auth_cookie
    params = {'userName': uid, 'password': password}
    res = http_session.post(f"https://{server}/User/DeepLensLogin", data=json.dumps(params))
    o = json.loads(await res.text())
    if o['code'] != 0:
        return False

    auth_cookie = res.headers["set-cookies"]
    return True


async def init_signalr():
    global signalr_conn, proctor_connections
    conn_builder = HubConnectionBuilder().with_url(hub_url)
    conn_builder.headers = {'Cookie': auth_cookie}  # Add cookie for user identification
    signalr_conn = conn_builder.build()

    async def camera_answer_from_taker(sdp):
        await test_taker_connection.setRemoteDescription(sdp)

    async def camera_answer_from_proctor(proctor, sdp):
        await proctor_connections[proctor].setRemoteDescription(sdp)

    async def camera_ice_candidate_from_taker(candidate):
        await test_taker_connection.addIceCandidate(candidate)

    async def camera_ice_candidate_from_proctor(proctor, candidate):
        await proctor_connections[proctor].addIceCandidate(candidate)

    signalr_conn.on("CameraAnswerFromTaker", camera_answer_from_taker)
    signalr_conn.on("CameraAnswerFromProctor", camera_answer_from_proctor)
    signalr_conn.on("CameraIceCandidateFromTaker", camera_ice_candidate_from_taker)
    signalr_conn.on("CameraIceCandidateFromProctor", camera_ice_candidate_from_proctor)

    await signalr_conn.start()


async def init_webrtc(proctors):
    test_taker_connection.addTrack(reader.video)
    taker_sdp = await test_taker_connection.createOffer()
    await signalr_conn.send("CameraOfferToTaker", taker_sdp)
    await test_taker_connection.setLocalDescription(taker_sdp)

    for proctor in proctors:
        conn = aiortc.RTCPeerConnection()
        conn.addTrack(reader.video)
        proctor_connections[proctor] = conn
        sdp = await conn.createOffer()
        await signalr_conn.send("CameraOfferToProctor", proctor, sdp)
        await conn.setLocalDescription(sdp)


def shutdown():
    test_taker_connection.close()
    for proctor in proctor_connections.keys():
        proctor_connections[proctor].close()

    signalr_conn.stop()


