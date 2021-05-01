import subprocess
import logging
import json
import re
import socket
import shlex
import os


BIOS_VERSION_PATH = '/sys/class/dmi/id/bios_version'

logger = logging.getLogger('SmartProctor-cam')


def execute(cmd, input_str=None, is_log=True, no_shlex=False):
    """Execute terminal commands"""

    if is_log:
        logger.debug('Command executing: ' + ' '.join(cmd))

    if input_str:
        proc = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE,
                                stdin=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        stdout = proc.communicate(input=input_str)[0]
    else:
        if no_shlex:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE)
        else:
            proc = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE)

        stdout = proc.stdout.decode('utf-8')

    if is_log:
        logger.debug(stdout)

    return proc.returncode, stdout


def is_network_connected(ssid):
    """Check if the device is connected to the suppied SSID"""

    stdout = execute('/usr/bin/nmcli -t -f name -e  no con show --active')[1]
    for line in stdout.splitlines():
        if line == ssid:
            return True

    return False


def is_network_inactive(ssid):
    """Check if the supplied SSID is current not connected, but previously
    connected"""

    is_present = False

    stdout = execute('/usr/bin/nmcli -t -f name -e no con show')[1]
    for line in stdout.splitlines():
        if line == ssid:
            is_present = True
            break

    if is_present and not is_network_connected(ssid):
        return True
    else:
        return False


def get_ip():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect(('114.114.114.114', 80))
    except:
        pass
    return s.getsockname()[0]


def get_network_status():
    """Get the current network status of the device"""

    stdout = execute('/usr/bin/nmcli -t -f name,type,device -e no con show --active')[1]

    status = {'ethernet': False, 'wifi': None}
    for line in stdout.splitlines():
        words = line.split(':')
        if len(words) < 2:
            continue

        # device name is wlp3s0 on linux PCs and mlan0 on DeepLens
        if words[1].find('wireless') > -1 and words[2] == 'mlan0':
            status['wifi'] = words[0]
        if words[1].find('ethernet') > -1:
            status['ethernet'] = True

    status['ip'] = get_ip()
    return status


def get_device_serial_number():
    """Obtain last 4 letters of Device Serial Number (DSN)"""

    (err, stdout) = execute('cat /sys/class/dmi/id/product_serial')
    return stdout.rstrip() if err == 0 else ''


def list_ssid():
    """Obtain the list of available Wi-Fi networks"""

    lines = execute(
        '/usr/bin/nmcli -t -f ssid,signal,security -e no device wifi list')[1].split('\n')

    ssid_dict = {}
    for l in lines:
        words = l.split(':')
        if len(words) > 3:
            words = [':'.join(words[0:-2]), words[-2], words[-1]]

        if words[0] == '--' or len(words[0]) == 0:
            continue

        signal_percent = float(words[1])
        bar = 1

        if signal_percent >= 90:
            bar = 4
        elif signal_percent >= 60:
            bar = 3
        elif signal_percent >= 30:
            bar = 2

        if words[0] in ssid_dict:
            bar = max(bar, ssid_dict[words[0]][0])

        ssid_dict[words[0]] = (bar, words[2])

    ssid_list = []
    for key, value in ssid_dict.items():
        ssid_list.append({
            'ssid': key,
            'strength': value[0],
            'security': value[1]
        })

    return ssid_list


def connect_wifi(wifi_name, wifi_password):
    """Connect the device to the provide Wi-Fi credentials"""

    if is_network_connected(wifi_name):
        return True

    if is_network_inactive(wifi_name):
        execute(['/usr/bin/nmcli', 'con', 'del', wifi_name], no_shlex=True)

    execute(['/usr/bin/nmcli', 'device', 'wifi', 'con', wifi_name,
                   'password', wifi_password, 'ifname', 'mlan0'], no_shlex=True)

    if not is_network_connected(wifi_name):
        execute(['/usr/bin/nmcli', 'con', 'del', wifi_name], no_shlex=True)
        return False

    return True
