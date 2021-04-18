import requests
import subprocess
import sys

required_package_installed = True
try:
    import aiortc
    import aiohttp
    import signalrcore
except ImportError:
    print('Required packages missing, attempting to install')
    required_package_installed = False


def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


if not required_package_installed:
    install_package('signalrcore')
    install_package('aiohttp')

    # The av package (referenced by aiortc) cannot be correctly installed, so manually download it from pypi.org
    r = requests.get('https://files.pythonhosted.org/packages/66/ff/bacde7314c646a2bd2f240034809a10cc3f8b096751284d0828640fff3dd/av-8.0.3-cp37-cp37m-manylinux2010_x86_64.whl')
    # Note that we should rename the file, otherwise 'manylinux2010' will not be acceptable by pip
    open('/tmp/av-8.0.3-cp37-cp37m-manylinux1_x86_64.whl', 'wb').write(r.content)
    install_package('/tmp/av-8.0.3-cp37-cp37m-manylinux1_x86_64.whl')

    r = requests.get('https://files.pythonhosted.org/packages/a2/24/9c2060d5f2d831091c7bb41428d17fd20e839959f1e78e6930329b21c0a7/pylibsrtp-0.6.8-cp37-cp37m-manylinux2010_x86_64.whl')
    open('/tmp/pylibsrtp-0.6.8-cp37-cp37m-manylinux1_x86_64.whl', 'wb').write(r.content)
    install_package('/tmp/pylibsrtp-0.6.8-cp37-cp37m-manylinux1_x86_64.whl')

    install_package('aiortc')
    print('Package installed')
