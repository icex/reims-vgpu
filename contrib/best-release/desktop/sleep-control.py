#!/usr/bin/python3
"""RAM pause and KDE window control; QEMU and its disks stay alive."""
import fcntl
import json
import os
from pathlib import Path
import runpy
import subprocess
import tempfile
import socket
import struct
import time

ICON = Path.home() / '.local/bin/reims-macos'


def window(pid, action):
    changes = {'minimize': 'w.minimized=true;',
               'show': 'w.minimized=false;workspace.activeWindow=w;',
               'progress': 'w.keepAbove=true;w.minimized=false;workspace.activeWindow=w;',
               'query': '', 'close': 'w.closeWindow();'}
    token = 'reims_sleep_' + str(time.time_ns())
    script = ('var ws=workspace.windowList().filter(w=>w.pid===' + str(int(pid)) + ');'
              'if(ws.length===1){var w=ws[0];' + changes[action] + '}'
              'print(' + json.dumps(token + ' ') + '+JSON.stringify(ws.map(w=>({pid:w.pid,'
              'minimized:w.minimized,caption:w.caption,id:String(w.internalId)}))));')
    with tempfile.TemporaryDirectory(prefix='reims-window-') as directory:
        path = Path(directory) / 'window.js'
        path.write_text(script)
        def call(*args):
            return subprocess.check_output(['qdbus6', 'org.kde.KWin', *args], text=True, timeout=3).strip()
        ident = call('/Scripting', 'org.kde.kwin.Scripting.loadScript', str(path), token)
        if int(ident) < 0:
            raise RuntimeError('KWin could not load the Reims window control.')
        try:
            call('/Scripting/Script' + ident, 'org.kde.kwin.Script.run')
        finally:
            call('/Scripting', 'org.kde.kwin.Scripting.unloadScript', token)
    deadline = time.monotonic() + 2
    while True:
        raw = subprocess.check_output(['journalctl', '--user', '-u', 'plasma-kwin_wayland.service',
                                      '--since', '1 minute ago', '--no-pager', '-o', 'cat'],
                                      text=True, timeout=2)
        lines = [line for line in raw.splitlines() if token + ' ' in line]
        if lines:
            windows = json.loads(lines[-1].split(token + ' ', 1)[1])
            if len(windows) != 1:
                raise RuntimeError('Expected one Reims window; found ' + str(len(windows)))
            result = windows[0]
            if action in ('show', 'minimize') and result['minimized'] != (action == 'minimize'):
                raise RuntimeError('KWin did not apply the Reims window state.')
            return result
        if time.monotonic() >= deadline:
            raise RuntimeError('KWin did not acknowledge the Reims window request.')
        time.sleep(.05)


def transition(run, action, manage_window=True):
    api = runpy.run_path(str(ICON))
    with (run / 'sleep-control.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with api['QMP'](run) as qmp:
            peer = qmp.connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
            pid = struct.unpack('3i', peer)[0]
            before = qmp.status()
            if action == 'pause':
                if before == 'running':
                    qmp.command('stop')
                elif before != 'paused':
                    raise RuntimeError('Cannot RAM-pause QEMU state ' + before)
                expected = 'paused'
            elif action == 'resume':
                if before == 'paused':
                    qmp.command('cont')
                elif before == 'suspended':
                    qmp.command('system_wakeup')
                elif before != 'running':
                    raise RuntimeError('Cannot resume QEMU state ' + before)
                expected = 'running'
            else:
                raise ValueError(action)
            deadline = time.monotonic() + 5
            while qmp.status() != expected:
                if time.monotonic() > deadline:
                    raise RuntimeError('QEMU did not reach ' + expected)
                time.sleep(.05)
        result = {'before': before, 'after': expected, 'pid': pid, 'action': action}
        if manage_window:
            try:
                result['window'] = window(pid, 'minimize' if action == 'pause' else 'show')
            except Exception as error:
                # A compositor failure must not discard or terminate the retained session.
                result['window_error'] = str(error)
        with (run / 'sleep-control.jsonl').open('a') as log:
            log.write(json.dumps({'time': time.time(), **result}) + '\n')
        return result
