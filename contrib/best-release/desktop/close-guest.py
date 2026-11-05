#!/usr/bin/python3
"""Native X retains the guest in RAM; logind sleep pauses a running guest."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
import runpy
import struct
import socket

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

control = module('sleep_control', HERE / 'sleep-control.py')
legacy = module('close_events', HERE / 'close-events.py')
saved = module('saved_machine', HERE / 'saved-machine.py')


class HostSleep:
    def __init__(self, run):
        import dbus
        from dbus.mainloop.glib import DBusGMainLoop
        from gi.repository import GLib
        DBusGMainLoop(set_as_default=True)
        self.loop = GLib.MainLoop()
        self.run = run
        self.resume_after_host = False
        self.saving = False
        self.fd = None
        self.bus = dbus.SystemBus()
        self.manager = dbus.Interface(self.bus.get_object('org.freedesktop.login1', '/org/freedesktop/login1'),
                                      'org.freedesktop.login1.Manager')
        self.match = self.bus.add_signal_receiver(self.prepare, signal_name='PrepareForSleep',
                        dbus_interface='org.freedesktop.login1.Manager',
                        bus_name='org.freedesktop.login1', path='/org/freedesktop/login1')
        self.inhibit()
        self.thread = threading.Thread(target=self.loop.run, daemon=True)
        self.thread.start()

    def inhibit(self):
        self.release()
        self.fd = self.manager.Inhibit('sleep', 'Reims macOS', 'Pause guest before host sleep', 'delay').take()

    def release(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def prepare(self, sleeping):
        try:
            if sleeping and self.saving:
                self.resume_after_host = False
                legacy.record('host-sleep-during-disk-save', guest_already_paused=True)
            elif sleeping:
                result = control.transition(self.run, 'pause', manage_window=False)
                self.resume_after_host = result['before'] == 'running'
                legacy.record('host-sleep-paused', **result)
            else:
                # Reacquire before resuming. A guest paused by X stays paused across host sleep.
                self.inhibit()
                if self.resume_after_host:
                    legacy.record('host-wake-resumed', **control.transition(self.run, 'resume', manage_window=False))
                self.resume_after_host = False
        except Exception as error:
            legacy.record('host-sleep-error', sleeping=bool(sleeping), error=repr(error))
            legacy.notify('Reims sleep handling failed: ' + str(error))
        finally:
            if sleeping:
                self.release()

    def close(self):
        self.match.remove()
        self.loop.quit()
        self.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--disk', type=Path)
    parser.add_argument('--ssh-port', type=int, required=True) # supervisor compatibility; no guest SSH
    parser.add_argument('--ssh-key', type=Path)
    parser.add_argument('--disk-save', action='store_true')
    opts = parser.parse_args()
    run = opts.run.resolve(strict=True)
    stopped = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopped.set())
    events = legacy.QMPEvents(run, opts.pid, opts.executable, stopped.is_set, disk=opts.disk)
    host = None
    shutdown = legacy.PowerdownHandler(opts.ssh_port, opts.ssh_key, run)
    try:
        host = HostSleep(run)
        legacy.record('ram-sleep-ready', pid=opts.pid)
        while not stopped.is_set():
            event = events.next()
            shutdown.tick()
            if not event:
                continue
            if event.get('event') == 'SHUTDOWN':
                break
            if event.get('event') == 'POWERDOWN':
                marker = run / 'explicit-shutdown.request'
                try:
                    requested = float(marker.read_text())
                    marker.unlink()
                except FileNotFoundError:
                    requested = 0
                if 0 <= time.time() - requested < 15:
                    shutdown.event(event)
                    continue
                if opts.disk_save:
                    api = runpy.run_path(str(control.ICON))
                    try:
                        with api['QMP'](run) as qmp:
                            peer = struct.unpack('3i', qmp.connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
                            legacy.validate_identity(run, opts.pid, opts.executable, peer, disk=opts.disk)
                            if qmp.status() != 'paused':
                                legacy.record('stale-native-close-ignored', state=qmp.status())
                                continue
                            host.saving = True
                            result = saved.save(run, qmp, opts.pid, legacy.notify)
                            legacy.record('native-close-saved-to-disk', image=result['image'], bytes=result['image_size'])
                        break
                    except Exception as error:
                        legacy.record('disk-save-error', error=repr(error))
                        legacy.notify('Disk save needs attention: ' + str(error) + '. Any committed saved image is preserved.')
                    finally:
                        host.saving = False
                else:
                    legacy.record('native-close-paused', **control.transition(run, 'pause'))
                    legacy.notify('macOS paused in RAM. Open the desktop shortcut to resume. Keep the host powered on.')
    except EOFError:
        pass
    finally:
        shutdown.close()
        if host:
            host.close()
        events.close()

if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        legacy.record('sleep-helper-error', error=repr(error))
        legacy.notify('Reims sleep helper failed. The VM remains running: ' + str(error))
        sys.exit(1)
