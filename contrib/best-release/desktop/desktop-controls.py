#!/usr/bin/env python3
"""Private SPICE clipboard client beside Reims' native Vulkan window."""
import argparse
import importlib.util
import json
import shutil
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
    wayland = bool(os.environ.get('WAYLAND_DISPLAY'))
    try:
        settings = json.loads(Path(__file__).with_name('desktop-settings.json').read_text())
    except (OSError, ValueError):
        settings = {}
    enabled = settings.get('clipboard', True) is not False and os.environ.get('REIMS_CLIPBOARD', '1') != '0'
    # GtkSession's focus-dependent clipboard and data-control must never both own it.
    clipboard.set_property('auto-clipboard', enabled and not wayland)
    bridge = worker = None
    bridge_timer = None
    bridge_module = None
    if wayland:
        if not all(shutil.which(tool) for tool in ('wl-copy', 'wl-paste')):
            enabled = False
            print('clipboard-disabled=missing-wayland-tools', flush=True)
        else:
            spec = importlib.util.spec_from_file_location('reims_wayland_clipboard', Path(__file__).with_name('wayland-clipboard.py'))
            bridge_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(bridge_module)
    clipboard.set_property('auto-usbredir', False)
    window = Gtk.Window(title='Reims — desktop controls')
    window.set_default_size(380, 160)
    box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
    box.set_border_width(14)
    label = Gtk.Label(label='Connecting private clipboard channel…')
    box.pack_start(label, False, False, 0)
    toggle = Gtk.CheckButton(label='Share text clipboard with Reims')
    toggle.set_active(enabled)
    def toggled(button):
        nonlocal enabled
        enabled = button.get_active()
        clipboard.set_property('auto-clipboard', enabled and not wayland)
        if bridge is not None:
            bridge.set_enabled(enabled)
        if worker is not None:
            worker.enable(enabled)
    toggle.connect('toggled', toggled)
    box.pack_start(toggle, False, False, 0)
    picker = None

    def pick_usb(*_):
        nonlocal picker
        if picker is None or picker.poll() is not None:
            picker = subprocess.Popen([str(Path(__file__).with_name('reims-usb-picker.sh'))], env=dict(os.environ, RGPU_RUN_DIR=str(run)))

    usb = Gtk.Button(label='USB devices…')
    usb.connect('clicked', pick_usb)
    box.pack_start(usb, False, False, 0)
    hint = Gtk.Label(label='Resize the Reims window normally.\nCtrl+Alt+Super releases keyboard capture.')
    box.pack_start(hint, False, False, 0)
    window.add(box)

    def channel_new(_, channel):
        nonlocal bridge, worker, bridge_timer
        if isinstance(channel, SpiceClientGLib.MainChannel):
            if bridge is not None:
                bridge.set_connected(False)
            if bridge_timer is not None:
                GLib.source_remove(bridge_timer)
            if worker is not None:
                worker.close()
            channel.set_property('max-clipboard', 65537)
            if bridge_module is not None:
                bridge = bridge_module.Clipboard(channel, lambda expected, value, generation: worker.write(expected, value, generation), lambda message: print(message, flush=True))
                last_error = [None]
                def failed(kind):
                    if kind != last_error[0]:
                        print('clipboard-backend-error=' + kind, flush=True)
                        last_error[0] = kind
                    return False
                worker = bridge_module.Worker(bridge_module.WaylandClipboard(), GLib.idle_add, bridge.host_changed, bridge.applied, failed)
                bridge.set_enabled(enabled)
                worker.enable(enabled)
                GObject.Object.connect(channel, 'main-clipboard-selection-request', bridge.requested)
                GObject.Object.connect(channel, 'main-clipboard-selection-grab', bridge.grabbed)
                GObject.Object.connect(channel, 'main-clipboard-selection-release', bridge.released)
                GObject.Object.connect(channel, 'main-clipboard-selection', bridge.received)
                bridge_timer = GLib.timeout_add(500, bridge.tick)
            def agent_changed(ch, _):
                connected = ch.get_property('agent-connected')
                label.set_text(('Clipboard connected (Wayland text)' if wayland else 'Clipboard connected') if connected else 'Waiting for the guest clipboard agent…')
                if bridge is not None:
                    bridge.set_connected(connected)
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
    last_request = None

    def usb_request():
        nonlocal last_request
        try:
            changed = (run / 'usb-picker.request').stat().st_mtime_ns
        except FileNotFoundError:
            return True
        if changed != last_request:
            last_request = changed
            pick_usb()
        return True

    GLib.timeout_add(250, connect)
    GLib.timeout_add(250, usb_request)
    window.show_all()
    try:
        Gtk.main()
    finally:
        if worker is not None:
            worker.close()
        session.disconnect()


if __name__ == '__main__':
    main()
