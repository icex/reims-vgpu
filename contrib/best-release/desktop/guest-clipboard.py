#!/usr/bin/env python3
"""Wait for guest Aqua, then install the dedicated clipboard once per revision."""
import argparse
import json
import os
import stat
from pathlib import Path
import shlex
import signal
import subprocess
import time

FILES = ('install.py', 'console-vdagent-agent.py', 'console-vdagent-clipboard.py',
         'console-vdagent-monitors.py', 'console-vdagent-handshake.py',
         'console-display-control.py', 'console-clipboard.m')
# The fixed command code receives only our explicit source files through stdin;
# no user path, clipboard text or credential is embedded in shell code/argv.
REMOTE = r'''
import json,os,pathlib,subprocess,sys,tempfile,shutil
files=json.load(sys.stdin)
allowed={'install.py','console-vdagent-agent.py','console-vdagent-clipboard.py','console-vdagent-monitors.py','console-vdagent-handshake.py','console-display-control.py','console-clipboard.m'}
if not isinstance(files,dict) or set(files)!=allowed or any(not isinstance(v,str) or len(v)>131072 for v in files.values()):raise ValueError('clipboard package refused')
os.umask(0o077)
stage=pathlib.Path(tempfile.mkdtemp(prefix='reims-clipboard-'))
try:
 for name,content in files.items():(stage/name).write_text(content)
 subprocess.run(['/usr/bin/python3','-B',str(stage/'install.py')],check=True,timeout=240)
finally:shutil.rmtree(stage)
'''
READY = '/usr/bin/python3 -c ' + shlex.quote('import os,subprocess; assert os.stat("/dev/console").st_uid==os.getuid(); subprocess.run(["launchctl","print","gui/"+str(os.getuid())],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=True)')


def ssh(port, key, known_hosts):
    return ['ssh', '-T', '-i', str(key), '-p', str(port), '-o', 'BatchMode=yes',
            '-o', 'ConnectTimeout=4', '-o', 'ControlMaster=no',
            '-o', 'StrictHostKeyChecking=yes', '-o', 'UserKnownHostsFile=' + str(known_hosts),
            'bogdan@127.0.0.1']


class StopBootstrap(Exception):
    pass


current_child = None


def run_command(argv, input=None, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=10):
    global current_child
    child = subprocess.Popen(argv, stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
                             stdout=stdout, stderr=stderr)
    current_child = child
    try:
        output, errors = child.communicate(input=input, timeout=timeout)
        return subprocess.CompletedProcess(argv, child.returncode, output, errors)
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=2)
        current_child = None


def stop(*_):
    if current_child is not None and current_child.poll() is None:
        current_child.terminate()
    raise StopBootstrap()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--ssh-port', type=int, required=True)
    parser.add_argument('--ssh-key', type=Path, required=True)
    parser.add_argument('--known-hosts', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    info = run.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise ValueError('private clipboard run directory required')
    source = Path(__file__).with_name('guest-clipboard')
    package = json.dumps({name: (source / name).read_text() for name in FILES}).encode()
    command = ssh(args.ssh_port, args.ssh_key, args.known_hosts)
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        try:
            ready = run_command(command + [READY], stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, timeout=10)
            if ready.returncode == 0:
                break
        except (OSError, subprocess.TimeoutExpired):
            pass
        time.sleep(2)
    else:
        raise RuntimeError('guest clipboard bootstrap: Aqua readiness timed out')
    for attempt in range(1, 4):
        try:
            result = run_command(command + ['/usr/bin/python3 -B -c ' + shlex.quote(REMOTE)],
                                    input=package, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                    timeout=270)
            if result.returncode == 0:
                # The installer prints source hashes and state only, never pasteboard text.
                outcome = json.loads(result.stdout)
                print(json.dumps(outcome), flush=True)
                return
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
        print('guest-clipboard-bootstrap-retry=' + str(attempt), flush=True)
        if attempt < 3:
            time.sleep(2)
    raise RuntimeError('guest clipboard bootstrap failed after three attempts')


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        main()
    except StopBootstrap:
        pass
