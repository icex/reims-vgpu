#!/usr/bin/env python3
"""Deliver native-window powerdown requests to an owned macOS guest over SSH."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import stat
import struct
import subprocess
import sys
import time

CONNECT_TIMEOUT = 30.0
QMP_TIMEOUT = 5.0
SSH_TIMEOUT = 40.0
PENDING_SECONDS = 60.0
NOTICE_SECONDS = 45.0
GUEST_SHUTDOWN = (
    'sudo -n shutdown -h now || { '
    'if pgrep -x Tailscale >/dev/null; then '
    'osascript -e \'tell application "Tailscale" to quit\'; fi; '
    'if pgrep -x Simulator >/dev/null; then '
    'osascript -e \'tell application "Simulator" to quit\'; fi; '
    'osascript -e \'tell application "System Events" to shut down\'; }'
)


def record(event, **values):
    print(json.dumps({'event': event, 'monotonic': time.monotonic(), **values}), flush=True)


def notify(message):
    try:
        subprocess.run(['notify-send', '-a', 'Reims macOS', '-i', 'computer',
                        'Reims macOS', message], timeout=5, check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired) as error:
        record('notification-error', error=repr(error))


def validate_identity(run, pid, executable, peer, proc_root=Path('/proc'), disk=None):
    """Bind the dedicated QMP peer to the exact child and its private disk."""
    if peer[0] != pid or peer[1] != os.getuid():
        raise RuntimeError('Unexpected close-events QMP pid or owner.')
    if run.stat().st_uid != os.getuid():
        raise RuntimeError('The Reims run is not owned by this account.')
    if (proc_root / str(pid) / 'exe').resolve(strict=True) != executable.resolve(strict=True):
        raise RuntimeError('Unexpected close-events QMP executable.')
    argv = (proc_root / str(pid) / 'cmdline').read_bytes().split(b'\0')
    disk = str(disk or run / 'macos.qcow2')
    drives = []
    for index, value in enumerate(argv[:-1]):
        if value == b'-drive':
            drives.extend(item.removeprefix('file=') for item in argv[index + 1].decode().split(',') if item.startswith('file='))
    if disk not in drives:
        raise RuntimeError('The close-events QMP peer does not use this run disk.')


class QMPEvents:
    def __init__(self, run, pid, executable, stop, proc_root=Path('/proc'), disk=None):
        self.socket = socket.socket(socket.AF_UNIX)
        self.socket.settimeout(0.5)
        self.buffer = b''
        self.queued = []
        endpoint = run / 'close-events.sock'
        deadline = time.monotonic() + CONNECT_TIMEOUT
        try:
            while not stop():
                try:
                    info = endpoint.lstat()
                    if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid():
                        raise RuntimeError('The close-events endpoint is not an owned socket.')
                    self.socket.connect(str(endpoint))
                    break
                except (FileNotFoundError, ConnectionRefusedError):
                    if time.monotonic() >= deadline:
                        raise RuntimeError('Timed out waiting for the dedicated close-events socket.')
                    time.sleep(0.1)
            else:
                raise RuntimeError('Close-events connection cancelled.')
            peer = struct.unpack('3i', self.socket.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
            validate_identity(run, pid, executable, peer, proc_root, disk)
            greeting = self.message(time.monotonic() + QMP_TIMEOUT)
            if not isinstance(greeting.get('QMP'), dict):
                raise RuntimeError('The close-events endpoint did not answer as QMP.')
            self.socket.sendall(b'{"execute":"qmp_capabilities","id":"close-events"}\n')
            deadline = time.monotonic() + QMP_TIMEOUT
            while True:
                reply = self.message(deadline)
                if 'event' in reply:
                    self.queued.append(reply)
                    continue
                if reply.get('id') != 'close-events' or 'return' not in reply or 'error' in reply:
                    raise RuntimeError('Close-events QMP capability negotiation failed.')
                break
            record('qmp-connected', pid=pid, uid=peer[1], executable=str(executable), endpoint=str(endpoint))
        except BaseException:
            self.socket.close()
            raise

    def message(self, deadline=None):
        while b'\n' not in self.buffer:
            if deadline is not None and time.monotonic() >= deadline:
                raise RuntimeError('Close-events QMP acknowledgement timed out.')
            try:
                chunk = self.socket.recv(65536)
            except socket.timeout:
                if deadline is None:
                    return None
                continue
            if not chunk:
                raise EOFError('Close-events QMP peer exited.')
            self.buffer += chunk
            if len(self.buffer) > 1024 * 1024:
                raise RuntimeError('Oversized close-events QMP response.')
        line, self.buffer = self.buffer.split(b'\n', 1)
        reply = json.loads(line)
        if not isinstance(reply, dict):
            raise RuntimeError('Invalid close-events QMP response.')
        return reply

    def next(self):
        return self.queued.pop(0) if self.queued else self.message()

    def close(self):
        self.socket.close()


def ssh_command(port, key, known_hosts):
    return ['ssh', '-i', str(key), '-p', str(port), '-o', 'BatchMode=yes',
            '-o', 'PasswordAuthentication=no', '-o', 'KbdInteractiveAuthentication=no',
            '-o', 'ConnectTimeout=4', '-o', 'StrictHostKeyChecking=accept-new',
            '-o', 'UserKnownHostsFile=' + str(known_hosts),
            '-o', 'ControlMaster=no', '-o', 'ControlPath=none',
            'bogdan@127.0.0.1', GUEST_SHUTDOWN]


def stop_transport(child):
    """Stop only the SSH transport created by this helper, never the VM."""
    if child is not None and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=2)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=2)


def retryable_transport_failure(returncode, stderr):
    """Retry an unavailable SSH transport, never an authentication failure."""
    if returncode != 255:
        return False
    error = stderr.lower()
    if any(marker in error for marker in (
            'permission denied', 'authentication failed', 'host key verification failed',
            'remote host identification has changed', 'no supported authentication')):
        return False
    return any(marker in error for marker in (
        'connection timed out', 'operation timed out', 'connection refused',
        'connection reset', 'connection closed', 'connection aborted'))


class PowerdownHandler:
    def __init__(self, port, key, run):
        self.port = port
        self.key = key
        self.known_hosts = run / 'close-known_hosts'
        self.child = None
        self.started = None
        self.pending = None
        self.notified = False
        self.period = 0
        self.attempts = 0
        self.retry_at = None

    def start_attempt(self):
        # Called only for a new POWERDOWN or a scheduled transport-only retry.
        # The QMP connection already pins this handler to its exact owned child.
        assert self.child is None
        self.pending = time.monotonic()
        self.notified = False
        self.retry_at = None
        self.attempts += 1
        command = ssh_command(self.port, self.key, self.known_hosts)
        try:
            self.child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self.started = time.monotonic()
            record('guest-shutdown-request', period=self.period, attempt=self.attempts,
                   ssh_pid=self.child.pid, ssh_port=self.port)
            if self.attempts == 1:
                notify('macOS shutdown requested. Complete any save dialogs in the guest.')
        except OSError as error:
            record('ssh-start-error', period=self.period, attempt=self.attempts, error=repr(error))
            notify('The macOS shutdown request failed. Use Apple menu > Shut Down; the VM remains running.')

    def event(self, event):
        if event.get('event') != 'POWERDOWN':
            return
        self.tick()
        if self.pending is not None:
            record('duplicate-powerdown-ignored', period=self.period)
            return
        self.period += 1
        self.attempts = 0
        self.start_attempt()

    def tick(self):
        now = time.monotonic()
        if self.child is not None:
            timeout = self.child.poll() is None and now - self.started >= SSH_TIMEOUT
            if timeout:
                stop_transport(self.child)
            if self.child.poll() is not None:
                stdout, stderr = self.child.communicate(timeout=2)
                returncode = self.child.returncode
                record('guest-shutdown-result', period=self.period, attempt=self.attempts,
                       returncode=returncode, timed_out=timeout, stdout=stdout, stderr=stderr)
                self.child = None
                # A running remote command may be waiting for a user's save
                # dialog. Never repeat it merely because SSH_TIMEOUT expired.
                if (not timeout and self.attempts < 3 and
                        retryable_transport_failure(returncode, stderr)):
                    self.retry_at = now + 30.0
                    record('guest-shutdown-retry-scheduled', period=self.period,
                           next_attempt=self.attempts + 1, delay_seconds=30)
                    notify('macOS is still starting or temporarily unreachable. Shutdown will retry automatically.')
                elif timeout or returncode:
                    notify('The macOS shutdown request did not finish. Complete guest dialogs or use Apple menu > Shut Down; the VM remains running.')
        if self.retry_at is not None and now >= self.retry_at:
            self.start_attempt()
        if self.pending is not None and not self.notified and now - self.pending >= NOTICE_SECONDS:
            notify('macOS is still running. Complete any save dialogs in the guest. Closing again will never force power off.')
            self.notified = True
        if (self.pending is not None and self.child is None and self.retry_at is None and
                now - self.pending >= PENDING_SECONDS):
            record('powerdown-pending-period-ended', period=self.period)
            self.pending = None

    def close(self):
        self.retry_at = None
        self.pending = None
        stop_transport(self.child)
        self.child = None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--disk', type=Path, help='exact guest disk; defaults to RUN/macos.qcow2')
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--ssh-port', type=int, required=True)
    parser.add_argument('--ssh-key', type=Path, default=Path.home() / 'macos-vm/vmkey')
    options = parser.parse_args(argv)
    if options.pid <= 1 or not 1 <= options.ssh_port <= 65535:
        parser.error('A valid child pid and SSH port are required.')
    run = options.run.resolve(strict=True)
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    handler = PowerdownHandler(options.ssh_port, options.ssh_key, run)
    events = None
    try:
        events = QMPEvents(run, options.pid, options.executable, lambda: stopped, disk=options.disk)
        while not stopped:
            event = events.next()
            if event is not None:
                if event.get('event') == 'SHUTDOWN':
                    record('guest-shutdown-event', data=event.get('data'))
                    break
                handler.event(event)
            handler.tick()
    except EOFError as error:
        record('qmp-disconnected', detail=str(error))
    finally:
        handler.close()
        if events is not None:
            events.close()


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        record('close-helper-error', error=repr(error))
        notify('The macOS close helper could not run. Use Apple menu > Shut Down; the VM remains running.')
        sys.exit(1)
