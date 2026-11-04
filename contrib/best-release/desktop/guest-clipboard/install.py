#!/usr/bin/env python3
"""Idempotent per-user Reims clipboard migration; keeps legacy files intact."""
import hashlib
import json
import os
from pathlib import Path
import plistlib
import re
import shutil
import stat
import subprocess
import time

FILES = ('console-vdagent-agent.py', 'console-vdagent-clipboard.py',
         'console-vdagent-monitors.py', 'console-vdagent-handshake.py',
         'console-display-control.py', 'console-clipboard.m')
LEGACY = 'org.raphaelgpu.clipboard'
LABEL = 'org.reims.clipboard'


def call(argv, check=False, timeout=15):
    return subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=timeout, check=check)


def private_dir(path):
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise ValueError('clipboard directory identity refused')
    path.chmod(0o700)


def atomic(path, value, mode=0o600):
    temporary = path.with_name('.' + path.name + '.new')
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, mode)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(value)
    temporary.chmod(mode)
    os.replace(temporary, path)


def loaded(domain, label):
    return call(['launchctl', 'print', domain + '/' + label]).returncode == 0


def disabled(domain, label):
    result = call(['launchctl', 'print-disabled', domain], check=True).stdout.decode()
    match = re.search(r'"' + re.escape(label) + r'"\s*=>\s*(true|false)', result)
    return None if match is None else match[1] == 'true'


def unload(domain, label):
    call(['launchctl', 'bootout', domain + '/' + label])
    for _ in range(30):
        if not loaded(domain, label):
            return
        time.sleep(.1)
    raise RuntimeError('clipboard service did not unload')


def install():
    if os.getuid() == 0 or os.getuid() != os.geteuid():
        raise ValueError('logged-in ordinary guest user required')
    os.umask(0o077)
    source = Path(__file__).resolve().parent
    root = Path.home() / 'Library/Application Support/Reims/clipboard-agent'
    private_dir(root)
    control = root / 'control'
    private_dir(control)
    domain = 'gui/' + str(os.getuid())
    call(['launchctl', 'print', domain], check=True)
    agents = Path.home() / 'Library/LaunchAgents'
    agents.mkdir(parents=True, exist_ok=True)
    plist = agents / (LABEL + '.plist')
    legacy_plist = agents / (LEGACY + '.plist')
    manifest = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in FILES}
    changed = False
    for name in FILES:
        content = (source / name).read_bytes()
        target = root / name
        if not target.is_file() or target.is_symlink() or target.read_bytes() != content:
            atomic(target, content)
            changed = True
    build_state = root / 'build.json'
    try:
        built = json.loads(build_state.read_text())
    except (OSError, ValueError):
        built = {}
    binary = root / 'console-clipboard'
    compiled = False
    if not binary.is_file() or binary.is_symlink() or built.get('source_sha256') != manifest['console-clipboard.m']:
        output = root / '.console-clipboard.new'
        # Refuse an unexpected link left at the temporary executable path.
        if output.exists() or output.is_symlink():
            output.unlink()
        call(['xcrun', 'clang', '-O2', '-fobjc-arc', str(root / 'console-clipboard.m'),
              '-framework', 'AppKit', '-o', str(output)], check=True, timeout=120)
        output.chmod(0o700)
        os.replace(output, binary)
        atomic(build_state, json.dumps({'source_sha256': manifest['console-clipboard.m']}).encode())
        compiled = True
    # Baseline read -1 never returns existing clipboard contents. Warm AppKit
    # before switching services, with bounded retries for a slow first launch.
    for attempt in range(3):
        warm = call([str(binary), 'read', '-1'], timeout=10)
        if warm.returncode == 0 and len(warm.stdout) == 12:
            break
        if attempt == 2:
            raise RuntimeError('clipboard helper warm-up failed')
        time.sleep(1)
    migration = root / 'migration.json'
    if not migration.exists():
        state = {'legacy_present': legacy_plist.is_file() or loaded(domain, LEGACY),
                 'legacy_loaded': loaded(domain, LEGACY),
                 'legacy_disabled': disabled(domain, LEGACY),
                 'reims_loaded': loaded(domain, LABEL),
                 'reims_disabled': disabled(domain, LABEL),
                 'reims_plist_present': plist.is_file()}
        if plist.is_file():
            atomic(root / 'previous-org.reims.clipboard.plist', plist.read_bytes())
        atomic(migration, json.dumps(state, indent=2).encode())
    # Only these two per-user clipboard labels are touched. Legacy source,
    # launcher and logs remain intact for removal/rollback.
    if legacy_plist.is_file() or loaded(domain, LEGACY):
        call(['launchctl', 'disable', domain + '/' + LEGACY], check=True)
        if loaded(domain, LEGACY):
            unload(domain, LEGACY)
    content = plistlib.dumps(dict(Label=LABEL, ProgramArguments=[
        '/usr/bin/python3', '-B', str(root / 'console-vdagent-agent.py'),
        '--control-dir', str(control), '--seconds', '0', '--guest-scale', '1', '--clipboard-text'],
        RunAtLoad=True, KeepAlive=True, ProcessType='Interactive', ThrottleInterval=10,
        LimitLoadToSessionType='Aqua', StandardOutPath=str(root / 'agent.log'),
        StandardErrorPath=str(root / 'agent-error.log')))
    plist_changed = not plist.is_file() or plist.read_bytes() != content
    if plist_changed:
        atomic(plist, content)
    restart = changed or compiled or plist_changed or not loaded(domain, LABEL)
    call(['launchctl', 'enable', domain + '/' + LABEL], check=True)
    if restart:
        if loaded(domain, LABEL):
            unload(domain, LABEL)
        call(['launchctl', 'bootstrap', domain, str(plist)], check=True)
    atomic(root / 'sources.json', json.dumps(manifest, sort_keys=True).encode())
    print(json.dumps({'clipboard_bootstrap': 'ok', 'restarted': restart,
                      'compiled': compiled, 'source_hashes': manifest}), flush=True)


if __name__ == '__main__':
    install()
