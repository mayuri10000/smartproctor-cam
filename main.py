import argparse
import asyncio
import json
import logging
import os
import platform
import ssl
from aiortc.codecs import h264

from aiohttp import web

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer


import app

app.start()

