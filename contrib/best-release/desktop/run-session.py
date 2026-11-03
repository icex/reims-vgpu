#!/usr/bin/env python3
"""Own only this Reims QEMU and its desktop controls, for one VM lifetime."""
import json
import re
import runpy
import fcntl
import time
import os
from pathlib import Path
import signal
import subprocess
import sys


def close_helper_args(run, args):
    """Build the dedicated helper command before launching the VM."""
    build = run / 'build-binary.json'
    if not build.is_file() or json.loads(build.read_text()).get('channel') != 'best':
        return None
    endpoint = 'unix:' + str(run / 'close-events.sock')
    if not any(arg == '-qmp' and index + 1 < len(args) and args[index + 1].split(',', 1)[0] == endpoint
               for index, arg in enumerate(args)):
        return None
    ports, disks = [], []
    for index, arg in enumerate(args[:-1]):
        if arg == '-netdev':
            ports += re.findall(r'(?:^|,)hostfwd=tcp:127\.0\.0\.1:([0-9]+)-:22(?:,|$)', args[index + 1])
        if arg == '-drive':
            disks += [part.removeprefix('file=') for part in args[index + 1].split(',')
                      if part.startswith('file=') and part.endswith('/macos.qcow2')]
    if len(ports) != 1 or len(disks) != 1:
        raise RuntimeError('The close helper needs one localhost SSH forward and one macOS disk.')
    helper = Path(__file__).with_name('close-guest.py')
    if not helper.is_file():
        raise RuntimeError('The Reims clean-close helper is missing; the VM was not started.')
    return [sys.executable, '-B', str(helper), '--run', str(run),
            '--executable', args[0], '--disk', disks[0], '--ssh-port', ports[0],
            '--ssh-key', str(Path.home() / 'macos-vm/vmkey')]


def stop_close_helper(child):
    # This is only the helper we spawned. It never owns or terminates QEMU.
    if child is not None and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=12)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=2)


def main():
    os.umask(0o077)
    run = Path(sys.argv[1]).resolve(strict=True)
    # Keep the existing coordination lease for the full interactive VM lifetime.
    lease = open('/var/tmp/reims-host-workload.lock', 'a')
    deadline = time.monotonic() + 10
    while True:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.monotonic() >= deadline:
                raise RuntimeError('Coordinated workload lease is busy; VM was not started')
            time.sleep(0.1)
    args = json.loads((run / 'command.json').read_text())
    if os.environ.get('REIMS_DAILY_SESSION') == '1':
        for key in list(os.environ):
            if key == 'REIMS_PERF' or key.startswith(('REIMS_VGPU_TRACE', 'REIMS_VGPU_FIFO_', 'REIMS_VGPU_GPU_')) or key in ('REIMS_VGPU_LOG', 'REIMS_VGPU_VRAM_SURFACES'):
                os.environ.pop(key, None)
        if json.loads((run / 'build-binary.json').read_text()).get('channel') == 'best':
            os.environ['REIMS_VGPU_VRAM_SURFACES'] = '1'
    if os.environ.get('REIMS_PERF'):
        # Profiling: QEMU runs as perf's child (the host's ptrace scope keeps perf from attaching
        # later), with sampling off until `enable` is written to the control fifo.
        ctl = run / 'perf.ctl'
        ctl.unlink(missing_ok=True)
        os.mkfifo(ctl, 0o600)
        # User-space software clock only: the host's perf_event_paranoid allows no more.
        args = ['perf', 'record', '-e', 'cpu-clock:u', '-D', '-1', f'--control=fifo:{ctl}',
                '-F', '1999', '-o', str(run / 'perf.data'), '--'] + args
    lan_fd = None
    if os.environ.get('REIMS_DAILY_SESSION') == '1' and os.environ.get('REIMS_LAN') == 'raphael':
        prepare = runpy.run_path(str(Path(__file__).with_name('prepare-lan.py')))['prepare']
        lan_fd, network = prepare()
        args += ['-netdev', f'tap,id=raphael_lan,fd={lan_fd}',
                 '-device', 'vmxnet3,netdev=raphael_lan,id=raphael_lan,mac=52:54:00:52:47:44,bus=pcie.0,addr=0x12']
        (run / 'runtime-network.json').write_text(json.dumps(network, indent=2) + '\n')
    helper_args = close_helper_args(run, args)
    qemu = subprocess.Popen(args, pass_fds=() if lan_fd is None else (lan_fd,))
    controls = None
    close_helper = None

    def stop(*_):
        if qemu.poll() is None:
            qemu.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        if helper_args is not None:
            with (run / 'close-guest.log').open('a') as log:
                try:
                    close_helper = subprocess.Popen(helper_args + ['--pid', str(qemu.pid)], stdout=log, stderr=log)
                except OSError as error:
                    log.write('Clean-close helper could not start: ' + repr(error) + '\n')
                    log.flush()
                    print('Clean-close helper unavailable; use the macOS Apple menu to shut down.', file=sys.stderr, flush=True)
        with (run / 'controls.log').open('a') as log:
            controls = subprocess.Popen([sys.executable, str(Path(__file__).with_name('desktop-controls.py')), str(run)], stdout=log, stderr=log)
            result = qemu.wait()
            (run / 'exit.json').write_text(json.dumps({'qemu_exit_code': result}) + '\n')
            return result
    finally:
        stop_close_helper(close_helper)
        if controls is not None and controls.poll() is None:
            controls.terminate()
            controls.wait(timeout=10)
        # These units belong only to Reims, never the running Raphael VM.
        for slot in (1, 2):
            subprocess.run(['systemctl', '--user', 'stop', f'reims-usbredir-{slot}.service'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == '__main__':
    sys.exit(main())
