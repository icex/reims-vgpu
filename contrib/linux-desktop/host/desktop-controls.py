#!/usr/bin/env python3
"""Private SPICE clipboard client beside Reims' native Vulkan window."""
import argparse
import os
from pathlib import Path
import signal
import stat
import subprocess
import gi

gi.require_version('Gtk', '3.0')
gi.require_version('SpiceClientGLib', '2.0')
gi.require_version('SpiceClientGtk', '3.0')
from gi.repository import Gtk, GLib, GObject, SpiceClientGLib, SpiceClientGtk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    info = run.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise RuntimeError('Private run directory required')
    session = SpiceClientGLib.Session()
    session.set_property('unix-path', str(run / 'spice.sock'))
    for name in ('enable-audio', 'enable-usbredir', 'enable-smartcard'):
        session.set_property(name, False)
    clipboard = SpiceClientGtk.GtkSession.get(session)
    clipboard.set_property('auto-clipboard', True)
    clipboard.set_property('auto-usbredir', False)
    window = Gtk.Window(title='Reims — desktop controls')
    window.set_default_size(380, 160)
    box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
    box.set_border_width(14)
    label = Gtk.Label(label='Connecting private clipboard channel…')
    box.pack_start(label, False, False, 0)
    toggle = Gtk.CheckButton(label='Share text clipboard with Reims')
    toggle.set_active(True)
    toggle.connect('toggled', lambda button: clipboard.set_property('auto-clipboard', button.get_active()))
    box.pack_start(toggle, False, False, 0)
    picker = None

    def pick_usb(*_):
        nonlocal picker
        if picker is None or picker.poll() is not None:
            picker = subprocess.Popen([str(Path(__file__).with_name('reims-usb-picker.sh'))], env=dict(os.environ, RGPU_RUN_DIR=str(run)))

    def reap_picker():
        nonlocal picker
        if picker is not None and picker.poll() is not None:
            picker = None
        return True

    usb = Gtk.Button(label='USB devices…')
    usb.connect('clicked', pick_usb)
    box.pack_start(usb, False, False, 0)
    hint = Gtk.Label(label='Resize the Reims window normally.\nCtrl+Alt+Super releases keyboard capture.')
    box.pack_start(hint, False, False, 0)
    window.add(box)

    def channel_new(_, channel):
        if isinstance(channel, SpiceClientGLib.MainChannel):
            def agent_changed(ch, _):
                connected = ch.get_property('agent-connected')
                label.set_text('Clipboard connected' if connected else 'Waiting for the guest clipboard agent…')
                print('clipboard-agent-connected=' + str(connected), flush=True)
            GObject.Object.connect(channel, 'notify::agent-connected', agent_changed)
            agent_changed(channel, None)
        # No display widget/channel is opened: Vulkan presents the native window.

    GObject.Object.connect(session, 'channel-new', channel_new)
    window.connect('delete-event', lambda widget, event: (widget.hide() or True))

    def connect():
        path = run / 'spice.sock'
        if not path.exists():
            return True
        info = path.lstat()
        if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid():
            raise RuntimeError('Invalid private SPICE socket')
        session.connect()
        return False

    signal.signal(signal.SIGTERM, lambda *_: GLib.idle_add(Gtk.main_quit))
    request = run / 'usb-picker.request'

    def request_mtime():
        try:
            return request.stat().st_mtime_ns
        except FileNotFoundError:
            return None

    # A request file left from before this process started does not open the picker.
    last_request = request_mtime()

    def usb_request():
        nonlocal last_request
        changed = request_mtime()
        if changed is None:
            return True
        if changed != last_request:
            last_request = changed
            pick_usb()
        return True

    GLib.timeout_add(250, connect)
    GLib.timeout_add(250, usb_request)
    GLib.timeout_add(1000, reap_picker)
    window.show_all()
    try:
        Gtk.main()
    finally:
        session.disconnect()


if __name__ == '__main__':
    main()
