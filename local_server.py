import argparse
import asyncio
import json
import logging
import os
import platform
import ssl
import proctor_client

import utils
import proctor_client

from aiohttp import web

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer
from video_reader import DeepLensVideoReader

ROOT = os.path.dirname(__file__)
CERT_DIR = 'certs/'


def add_cors_header(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-CSRF-Token, Cache-Control'


async def serial_number(request):
    res = web.Response(
        content_type="application/json",
        text=json.dumps(
            {"serialNumber": utils.get_device_serial_number()}
        ),
    )
    add_cors_header(res)
    return res


async def wifi_ssids(request):
    res = web.Response(
        content_type="application/json",
        text=json.dumps(
            {'wifiList': utils.list_ssid()}
        ),
    )
    add_cors_header(res)
    return res


async def connect_wifi(request: web.Request):
    params = request.query
    res = web.Response(
        content_type="application/json",
        text=json.dumps(
            {'success': utils.connect_wifi(params['ssid'], params['password'])}
        ),
    )
    add_cors_header(res)
    return res


async def network_status(request):
    res = web.Response(
        content_type="application/json",
        text=json.dumps(
            utils.get_network_status()
        ),
    )
    add_cors_header(res)
    return res


async def login(request):
    params = request.query
    login_res = await proctor_client.login(params['token'], params['eid'])
    if login_res:
        await proctor_client.init_exam()
    res = web.Response(
        content_type="application/json",
        text=json.dumps(
            {"success": login_res}
        ),
    )
    add_cors_header(res)
    return res


def on_shutdown():
    proctor_client.shutdown()


def start():
    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/login", login)
    app.router.add_get('/network_status', network_status)
    app.router.add_get('/connect_wifi', connect_wifi)
    app.router.add_get('/wifi_ssids', wifi_ssids)
    app.router.add_get('/sn', serial_number)

    ssl_context = ssl.SSLContext()
    ssl_context.load_cert_chain(CERT_DIR + 'fullchain.pem', CERT_DIR + 'privkey.pem')

    web.run_app(app, host="0.0.0.0", port=8080, ssl_context=ssl_context)
