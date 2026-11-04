#!/usr/bin/env python3
"""A loopback-only, per-Reims-lifetime home share; failures are optional services.

Passwords are read only from a private host file and transported to Samba and
macOS on standard input. Never print subprocess payloads or credential contents.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import plistlib
import secrets
import shlex
import signal
import socket
import stat
import subprocess
import threading
import time

IMAGE = 'raphael-macos-shares:local'
LABEL = 'org.reims.home-share'
ACCOUNT = 'macosshare'
SHARE = 'ReimsHome'
PORT = 4450
IMAGE_SUFFIXES = {'.qcow2', '.qcow', '.img', '.raw', '.vmdk', '.vdi', '.dmg', '.iso'}
CONTAINER_SCRIPT = '''#!/bin/sh
set -eu
[ "$(id -u macosshare)" = 1000 ]
[ "$(id -g macosshare)" = 1000 ]
mkdir -p /var/lib/samba/private /var/cache/samba /run/samba /var/log/samba
password=$(cat /reims-config/password)
printf '%s\\n%s\\n' "$password" "$password" | smbpasswd -s -a macosshare >/dev/null 2>&1
unset password
exec /usr/sbin/smbd --foreground --no-process-group --debug-stdout --configfile=/reims-config/smb.conf
'''


def private_directory(path):
    path = Path(path)
    path.mkdir(parents=True, mode=0o700, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise RuntimeError('Home-share state directory must be owned by this user')
    path.chmod(0o700)
    return path


def private_write(path, content):
    path = Path(path)
    # Refuse symlink substitution; every parent is a private, user-owned directory.
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(content.encode() if isinstance(content, str) else content)


def protected_images(home, run=None):
    """Protect existing images, including their aliases through symlinks in home.

    Samba's veto list matches components. Exact image basenames are therefore
    excluded throughout this share, including duplicate names outside macos-vm.
    The container also masks each real image read-only with an empty file.
    """
    root = Path(home) / 'macos-vm'
    images = []
    if root.exists():
        for folder, dirs, files in os.walk(root, followlinks=False):
            for name in files:
                if Path(name).suffix.lower() in IMAGE_SUFFIXES:
                    path = Path(folder) / name
                    if path.is_file():
                        real = path.resolve(strict=True)
                        if real.is_relative_to(Path(home).resolve()):
                            images.append(real)
    if run is not None:
        run = Path(run).resolve(strict=True)
        command_file = run / 'command.json'
        if command_file.is_file():
            command = json.loads(command_file.read_text())
            if not isinstance(command, list) or not all(isinstance(a, str) for a in command):
                raise ValueError('Invalid VM command metadata')
            for index, arg in enumerate(command[:-1]):
                if arg != '-drive':
                    continue
                for part in command[index + 1].split(','):
                    if part.startswith('file='):
                        path = Path(part[5:]).resolve(strict=True)
                        if not path.is_relative_to(Path(home).resolve()):
                            raise ValueError('VM storage must be inside the exported home')
                        if path.is_file():
                            images.append(path)
        # The disposable runner keeps its ONE pristine snapshot beside run logs.
        pristine = run.parent / 'pristine'
        if pristine.is_dir():
            for path in pristine.iterdir():
                if path.is_file() and (path.suffix.lower() in IMAGE_SUFFIXES or path.name == 'OVMF_VARS.fd'):
                    path = path.resolve(strict=True)
                    if not path.is_relative_to(Path(home).resolve()):
                        raise ValueError('Pristine storage must be inside the exported home')
                    images.append(path)
    return sorted(set(images))


def samba_config(home, images, port=PORT, encryption='if_required'):
    home = Path(home).resolve()
    names = sorted({'reims-9070xt', 'macos.qcow2', '.DS_Store', '._*', *(p.name for p in images)})
    if any(any(c in n for c in '\n\r/%') for n in names):
        raise ValueError('Unsupported protected image basename')
    if encryption not in ('if_required', 'required'):
        raise ValueError('SMB encryption must be if_required or required')
    return f'''[global]
server role = standalone server
workgroup = WORKGROUP
server string = Reims private home
interfaces = 127.0.0.1/32
bind interfaces only = yes
smb ports = {int(port)}
disable netbios = yes
server min protocol = SMB3_00
server max protocol = SMB3_11
server signing = mandatory
server smb encrypt = {encryption}
security = user
ntlm auth = ntlmv2-only
map to guest = Never
restrict anonymous = 2
load printers = no
printing = bsd
printcap name = /dev/null
disable spoolss = yes
usershare allow guests = no
unix extensions = no
smb3 unix extensions = no
private dir = /var/lib/samba/private
state directory = /var/lib/samba
cache directory = /var/cache/samba
lock directory = /run/samba
pid directory = /run/samba
log level = 0
logging = stdout
[ReimsHome]
path = {home}
browseable = yes
read only = no
guest ok = no
valid users = {ACCOUNT}
force user = {ACCOUNT}
force group = {ACCOUNT}
create mask = 0666
directory mask = 0777
wide links = no
follow symlinks = yes
veto files = /{'/'.join(names)}/
delete veto files = yes
hide dot files = yes
vfs objects = fruit streams_xattr
fruit:metadata = stream
fruit:resource = stream
fruit:encoding = native
fruit:time machine = no
fruit:zero_file_id = yes
spotlight = no
map archive = no
map hidden = no
map system = no
store dos attributes = yes
'''


def notify(message):
    try:
        subprocess.run(['notify-send', 'Reims home share', message],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        pass


class HomeShare:
    def __init__(self, run, *, home=None, state=None, port=PORT, image=IMAGE):
        self.run = Path(run).resolve(strict=True)
        self.home = Path(home or Path.home()).resolve(strict=True)
        if os.getuid() != 1000:
            raise RuntimeError('This integration requires the desktop owner UID 1000')
        self.state = Path(state or self.home / '.config/reims-macos/home-share')
        self.port, self.image = int(port), image
        self.container = None
        self.stop_event = threading.Event()
        digest = hashlib.sha256(str(self.run).encode()).hexdigest()[:16]
        self.name = 'reims-home-1000-' + digest
        self.owner = digest

    def docker(self, *args, check=True, timeout=40):
        result = subprocess.run(['docker', *args], capture_output=True,
                                text=True, timeout=timeout)
        if check and result.returncode:
            # Docker arguments and captured output contain no credentials, but
            # avoid echoing either: they may include unrelated daemon details.
            raise RuntimeError('Docker home-share operation failed: ' + args[0])
        return result

    def start(self, encryption='if_required'):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as admission:
            admission.bind(('127.0.0.1', self.port))
        private_directory(self.state)
        for name in ('masked-storage',):
            private_directory(self.state / name)
        credential = self.state / 'password'
        if not credential.exists():
            fd = os.open(credential, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, 'w') as stream:
                stream.write(secrets.token_hex(32) + '\n')
        info = credential.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise RuntimeError('Invalid private home-share credential file')
        credential.chmod(0o600)
        password = credential.read_text().strip()
        if len(password) != 64 or any(c not in '0123456789abcdef' for c in password):
            raise RuntimeError('Invalid private home-share credential format')
        images = protected_images(self.home, self.run)
        private_write(self.state / 'smb.conf', samba_config(self.home, images, self.port, encryption))
        private_write(self.state / 'start.sh', CONTAINER_SCRIPT)
        private_write(self.state / 'masked-image', b'')
        existing = self.docker('inspect', self.name, check=False)
        if existing.returncode == 0:
            row = json.loads(existing.stdout)[0]
            if row['Config'].get('Labels', {}).get(LABEL) != self.owner:
                raise RuntimeError('Refusing to replace an unrelated container')
            # A duplicate lifetime must never stop the live owner's share.
            raise RuntimeError('A home-share container already exists for this run')
        # No publishing or Docker bridge exists: smbd itself binds host loopback.
        command = ['run', '--detach', '--rm', '--name', self.name,
                   '--label', LABEL + '=' + self.owner, '--network', 'host',
                   '--read-only', '--security-opt', 'no-new-privileges:true',
                   '--cap-drop', 'NET_RAW', '--tmpfs', '/tmp', '--tmpfs', '/run/samba',
                   '--tmpfs', '/var/cache/samba', '--tmpfs', '/var/lib/samba',
                   '--tmpfs', '/var/log/samba',
                   '--mount', f'type=bind,src={self.state},dst=/reims-config,readonly',
                   '--mount', f'type=bind,src={self.home},dst={self.home}']
        protected = self.home / 'reims-9070xt'
        if protected.exists():
            command += ['--mount', f'type=bind,src={self.state}/masked-storage,dst={protected},readonly']
        for path in images:
            if path.is_relative_to(protected):
                continue  # Already covered by the empty read-only directory.
            command += ['--mount', f'type=bind,src={self.state}/masked-image,dst={path},readonly']
        command += ['--entrypoint', '/bin/sh', self.image, '/reims-config/start.sh']
        if self.stop_event.is_set():
            raise RuntimeError('Home-share startup was cancelled')
        self.container = self.name  # Recover an owned container even if docker run times out.
        self.container = self.docker(*command).stdout.strip()
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if self.stop_event.is_set():
                raise RuntimeError('Home-share startup was cancelled')
            status = self.docker('inspect', '--format', '{{.State.Running}}', self.container, check=False)
            if status.returncode or status.stdout.strip() != 'true':
                raise RuntimeError('Home-share server did not stay running')
            try:
                with socket.create_connection(('127.0.0.1', self.port), timeout=1):
                    return {'container': self.container, 'name': self.name,
                            'endpoint': f'127.0.0.1:{self.port}', 'share': SHARE,
                            'encryption': encryption, 'protected_images': [str(p) for p in images],
                            'vetoed_image_basenames': sorted({p.name for p in images}),
                            'masked_directory': str(protected)}
            except OSError:
                time.sleep(.25)
        raise RuntimeError('Home-share server did not accept a loopback connection')

    def stop(self):
        self.stop_event.set()
        if self.container is not None:
            row = self.docker('inspect', self.container, check=False)
            if row.returncode == 0:
                info = json.loads(row.stdout)[0]
                if info['Config'].get('Labels', {}).get(LABEL) != self.owner:
                    raise RuntimeError('Refusing to stop an unrelated container')
                stopped = self.docker('stop', '--timeout', '10', self.container, check=False, timeout=20)
                if stopped.returncode:
                    raise RuntimeError('Home-share container did not stop')
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    if self.docker('inspect', self.container, check=False).returncode:
                        break
                    time.sleep(.1)
                else:
                    raise RuntimeError('Home-share container removal timed out')
            self.container = None

    def bootstrap(self, ssh_args, *, wait_seconds=360):
        """Run in a worker thread after QEMU starts; no exception escapes its caller.

        ssh_args is the full trusted argv through user@host. The installer and
        new password travel only on SSH stdin. Returned evidence never contains
        the installer payload or subprocess output.
        """
        deadline = time.monotonic() + wait_seconds
        while not self.stop_event.is_set() and time.monotonic() < deadline:
            try:
                check = subprocess.run([*ssh_args, 'test "$(stat -f%Su /dev/console)" = "$USER" && pgrep -x Finder >/dev/null'],
                                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL, timeout=12)
                if check.returncode == 0:
                    password = (self.state / 'password').read_text().strip()
                    payload = guest_installer(password, self.port)
                    result = subprocess.run([*ssh_args, '/bin/sh -s'], input=payload,
                                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                            timeout=60, text=True)
                    if result.returncode == 0:
                        return {'installed': True, 'share': SHARE}
            except (OSError, subprocess.TimeoutExpired):
                pass
            self.stop_event.wait(5)
        return {'installed': False, 'stopped': self.stop_event.is_set()}


def guest_agent(port=PORT):
    # mount volume is an AppleScript command using an in-memory password.
    # security's password stdout goes only into AppleScript, never launchd logs.
    return f'''#!/bin/sh
set -eu
/sbin/mount | /usr/bin/grep -F ' on /Volumes/{SHARE} ' >/dev/null && exit 0
/usr/bin/nc -z -G 2 -w 2 10.0.2.2 {int(port)} >/dev/null 2>&1 || exit 0
/usr/bin/osascript >/dev/null 2>&1 <<'APPLESCRIPT'
try
    set sharePassword to do shell script "/usr/bin/security find-internet-password -s 10.0.2.2 -a {ACCOUNT} -p {SHARE} -P {int(port)} -r 'smb ' -w"
    mount volume "smb://10.0.2.2:{int(port)}/{SHARE}" as user name "{ACCOUNT}" with password sharePassword
end try
APPLESCRIPT
'''


def guest_installer(password, port=PORT):
    if len(password) != 64 or any(c not in '0123456789abcdef' for c in password):
        raise ValueError('Invalid credential format')
    support = 'Library/Application Support/Reims/home-share'
    agent = guest_agent(port)
    plist = plistlib.dumps({'Label': 'org.reims.home-share',
                           'ProgramArguments': ['/bin/sh', '/Users/bogdan/' + support + '/mount.sh'],
                           'RunAtLoad': True, 'StartInterval': 15,
                           'ProcessType': 'Background',
                           'StandardOutPath': '/dev/null', 'StandardErrorPath': '/dev/null'})
    # The old agent's full installer/config is archived on the guest once; its
    # keychain items are untouched. Only the named obsolete agent is unloaded.
    return f'''set -eu
umask 077
support="$HOME/{support}"
mkdir -p "$support" "$HOME/Library/LaunchAgents"
if [ ! -e "$support/before.json" ]; then
    /usr/bin/defaults read com.apple.desktopservices DSDontWriteNetworkStores > "$support/network-stores.before" 2>/dev/null || :
    if [ -f "$HOME/Library/LaunchAgents/org.raphael.linux-shares.plist" ]; then
        cp -p "$HOME/Library/LaunchAgents/org.raphael.linux-shares.plist" "$support/org.raphael.linux-shares.plist.before"
    fi
    printf '{{"version":1}}\\n' > "$support/before.json"
fi
/usr/bin/launchctl bootout "gui/$(id -u)/org.raphael.linux-shares" >/dev/null 2>&1 || :
if [ -f "$HOME/Library/LaunchAgents/org.raphael.linux-shares.plist" ]; then
    mv "$HOME/Library/LaunchAgents/org.raphael.linux-shares.plist" "$support/org.raphael.linux-shares.plist.disabled"
fi
/usr/bin/security -i >/dev/null 2>&1 <<KEYCHAIN
add-internet-password -U -a {ACCOUNT} -s 10.0.2.2 -p {SHARE} -P {int(port)} -r 'smb ' -l 'Reims Linux home' -T /usr/bin/security -T /usr/bin/osascript -w {password} "$HOME/Library/Keychains/login.keychain-db"
KEYCHAIN
/usr/bin/defaults write com.apple.desktopservices DSDontWriteNetworkStores -bool true
printf %s {shlex.quote(base64.b64encode(agent.encode()).decode())} | /usr/bin/base64 -D > "$support/mount.sh"
chmod 700 "$support/mount.sh"
printf %s {shlex.quote(base64.b64encode(plist).decode())} | /usr/bin/base64 -D > "$HOME/Library/LaunchAgents/org.reims.home-share.plist"
/usr/bin/launchctl bootout "gui/$(id -u)/org.reims.home-share" >/dev/null 2>&1 || :
/usr/bin/launchctl bootstrap "gui/$(id -u)" "$HOME/Library/LaunchAgents/org.reims.home-share.plist"
if [ ! -e "$HOME/Desktop/Linux Home" ] && [ ! -L "$HOME/Desktop/Linux Home" ]; then
    ln -s /Volumes/{SHARE} "$HOME/Desktop/Linux Home"
fi
/bin/sh "$support/mount.sh"
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--home', type=Path)
    parser.add_argument('--state', type=Path)
    parser.add_argument('--port', type=int, default=PORT)
    parser.add_argument('--ssh-port', type=int, default=10023)
    parser.add_argument('--ssh-key', type=Path, default=Path.home() / 'macos-vm/vmkey')
    parser.add_argument('--known-hosts', type=Path, default=Path.home() / 'reims-9070xt/known_hosts')
    parser.add_argument('--encryption', choices=('if_required', 'required'), default='if_required')
    args = parser.parse_args()
    service = HomeShare(args.run, home=args.home, state=args.state, port=args.port)
    signal.signal(signal.SIGTERM, lambda *_: service.stop_event.set())
    signal.signal(signal.SIGINT, lambda *_: service.stop_event.set())
    try:
        metadata = service.start(args.encryption)
        print(json.dumps(metadata), flush=True)
        ssh = ['ssh', '-i', str(args.ssh_key), '-p', str(args.ssh_port),
               '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=4',
               '-o', 'StrictHostKeyChecking=yes', '-o', 'ControlMaster=no',
               '-o', 'UserKnownHostsFile=' + str(args.known_hosts), 'bogdan@127.0.0.1']
        installed = service.bootstrap(ssh)
        print(json.dumps(installed), flush=True)
        if not installed['installed'] and not installed.get('stopped'):
            notify('Guest setup did not finish. The VM can continue normally.')
        service.stop_event.wait()
        return 0
    except Exception as error:
        print('Home share unavailable: ' + str(error), flush=True)
        notify('Home share unavailable. The VM can continue normally.')
        return 1
    finally:
        service.stop()


if __name__ == '__main__':
    raise SystemExit(main())
