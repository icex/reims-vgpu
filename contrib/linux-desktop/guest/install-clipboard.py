#!/usr/bin/env python3
"""Install only Reims' text clipboard agent in this guest user session."""
import os
from pathlib import Path
import plistlib
import shutil
import subprocess

# The agent and the modules it loads beside itself (console-vdagent-agent.py loads the
# monitors, display-control and clipboard modules; the monitors module loads the
# handshake module), and the pasteboard helper's source, which is compiled below.
AGENT_FILES = (
    'console-vdagent-agent.py',
    'console-vdagent-monitors.py',
    'console-vdagent-handshake.py',
    'console-display-control.py',
    'console-vdagent-clipboard.py',
    'console-clipboard.m',
)


def main():
    if os.geteuid() == 0:
        raise SystemExit('Run as the logged-in guest user')
    root = Path.home() / 'Library/Application Support/Reims'
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o700)
    control = root / 'control'
    control.mkdir(exist_ok=True)
    control.chmod(0o700)
    source = Path(__file__).resolve().parent
    for name in AGENT_FILES:
        path = source / name
        target = root / name
        if target != path:
            shutil.copy2(path, target)
    subprocess.run(['xcrun', 'clang', '-O2', '-fobjc-arc', str(root / 'console-clipboard.m'),
        '-framework', 'AppKit', '-o', str(root / 'console-clipboard')], check=True)
    (root / 'console-clipboard').chmod(0o700)
    agent = Path.home() / 'Library/LaunchAgents/org.reims.clipboard.plist'
    agent.parent.mkdir(parents=True, exist_ok=True)
    label = f'gui/{os.getuid()}'
    subprocess.run(['launchctl', 'bootout', label + '/org.reims.clipboard'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    agent.write_bytes(plistlib.dumps(dict(Label='org.reims.clipboard',
        ProgramArguments=['/usr/bin/python3', '-B', str(root / 'console-vdagent-agent.py'),
            '--control-dir', str(control), '--seconds', '0', '--guest-scale', '1', '--clipboard-text'],
        RunAtLoad=True, KeepAlive=True, ProcessType='Interactive', ThrottleInterval=10,
        StandardOutPath=str(root / 'clipboard.log'), StandardErrorPath=str(root / 'clipboard-error.log'))))
    agent.chmod(0o600)
    subprocess.run(['launchctl', 'bootstrap', label, str(agent)], check=True)
    print('Reims text clipboard agent installed for this guest user.')


if __name__ == '__main__':
    main()
