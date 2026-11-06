#!/usr/bin/python3
"""Friendly USB picker labels from sysfs; never opens or claims a USB device."""
from pathlib import Path
import sys


def describe(bus, device, fallback, root=Path('/sys/bus/usb/devices')):
    def read(path):
        try:
            return path.read_text().strip()
        except OSError:
            return ''
    try:
        nodes = list(root.iterdir())
    except OSError:
        return fallback
    for node in nodes:
        if read(node / 'busnum') != str(int(bus)) or read(node / 'devnum') != str(int(device)):
            continue
        product = read(node / 'product') or fallback
        classes = [(read(i / 'bInterfaceClass').lower(),
                    read(i / 'bInterfaceSubClass').lower(),
                    read(i / 'bInterfaceProtocol').lower())
                   for i in root.glob(node.name + ':*')]
        if any(c == '08' for c, _, _ in classes):
            return 'USB storage — ' + product
        if ('e0', '01', '01') in classes:
            return 'Bluetooth adapter — ' + product
        return product
    return fallback


if __name__ == '__main__':
    print(describe(*sys.argv[1:4]))
