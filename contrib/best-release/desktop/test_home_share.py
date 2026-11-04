#!/usr/bin/env python3
"""No VM, no Samba, no real home, and no live credentials in these tests."""
import importlib.util
import ast
import base64
import json
import re
import threading
import types
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('home_share', ROOT / 'home-share.py')
share = importlib.util.module_from_spec(spec)
spec.loader.exec_module(share)


class ShareTests(unittest.TestCase):
    def test_private_modes_and_symlink_refusal(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root = share.private_directory(Path(temp) / 'state')
            self.assertEqual(root.stat().st_mode & 0o777, 0o700)
            p = root / 'password'
            share.private_write(p, 'fixture')
            self.assertEqual(p.stat().st_mode & 0o777, 0o600)
            link = root / 'alias'
            link.symlink_to(p)
            with self.assertRaises(OSError):
                share.private_write(link, 'bad')
            self.assertEqual(p.read_text(), 'fixture')

    def test_protected_images_and_aliases(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            home = Path(temp)
            disks = home / 'macos-vm'
            disks.mkdir()
            (disks / 'macos.qcow2').write_text('disk fixture')
            (disks / 'notes.txt').write_text('notes')
            (disks / 'another.img').symlink_to(disks / 'macos.qcow2')
            images = share.protected_images(home)
            self.assertEqual(images, [disks / 'macos.qcow2'])
            config = share.samba_config(home, images)
            self.assertIn('/macos.qcow2/', config)
            self.assertNotIn('notes.txt', config)
            self.assertIn('interfaces = 127.0.0.1/32', config)
            self.assertIn('smb ports = 4450', config)
            self.assertIn('bind interfaces only = yes', config)
            self.assertIn('force user = macosshare', config)
            self.assertIn('wide links = no', config)
            self.assertIn('delete veto files = yes', config)
            self.assertIn('vfs objects = fruit streams_xattr', config)
            self.assertIn('spotlight = no', config)

    def test_active_and_pristine_disposable_disks_are_protected(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            home = Path(temp)
            for name in ('disk', 'pristine', 'run'):
                (home / name).mkdir()
            for name in ('macos.qcow2', 'OpenCore.qcow2', 'OVMF_VARS.fd'):
                for folder in ('disk', 'pristine'):
                    (home / folder / name).write_text('protected fixture')
            command = ['/fixture/qemu']
            for path in (home / 'disk').iterdir():
                command += ['-drive', 'format=raw,file=' + str(path)]
            (home / 'run/command.json').write_text(__import__('json').dumps(command))
            images = share.protected_images(home, home / 'run')
            self.assertEqual(len(images), 6)
            self.assertIn(home / 'disk/macos.qcow2', images)
            self.assertIn(home / 'pristine/macos.qcow2', images)
            self.assertIn('/macos.qcow2/', share.samba_config(home, []))

    def test_guest_credential_only_stdin_not_persistent_files(self):
        secret = 'a' * 64
        payload = share.guest_installer(secret)
        self.assertEqual(payload.count(secret), 1)
        self.assertIn('mkfifo -m 600', payload)
        self.assertNotIn('<<KEYCHAIN\n', payload)
        writer = next(line for line in payload.splitlines() if secret in line)
        self.assertEqual(writer, 'printf %s ' + secret + ' > \"$keychain_temp/input\" &')
        assets = re.findall(r'^printf %s ([A-Za-z0-9+/=]+) \| /usr/bin/base64 -D', payload, re.M)
        self.assertEqual(len(assets), 3)
        worker = base64.b64decode(assets[0]).decode()
        ast.parse(worker)
        self.assertIn('login.keychain-db', worker)
        self.assertNotIn(secret, worker)
        self.assertIn("subprocess.run(['/usr/bin/security', '-i'], input=", worker)
        self.assertNotIn(secret, base64.b64decode(assets[1]).decode())
        self.assertEqual(__import__('plistlib').loads(base64.b64decode(assets[2]))['StartInterval'], 15)
        agent = share.guest_agent()
        self.assertNotIn(secret, agent)
        self.assertIn('find-internet-password', agent)
        self.assertIn('with password sharePassword', agent)
        shell = subprocess.run(['/bin/sh', '-n'], input=payload, text=True, capture_output=True)
        self.assertEqual(shell.returncode, 0, shell.stderr)
        self.assertEqual(subprocess.run(['/bin/sh', '-n'], input=agent, text=True).returncode, 0)

    def test_shortcut_is_outside_icloud_desktop(self):
        payload = share.guest_installer('d' * 64)
        self.assertIn('"$HOME/Linux Home"', payload)
        self.assertNotIn('$HOME/Desktop', payload)
        self.assertNotIn('$HOME/Documents', payload)

    def test_gui_worker_keeps_credential_in_security_stdin(self):
        namespace = {'__name__': 'keychain_fixture'}
        exec(compile(share.guest_keychain_worker(), '<fixture-worker>', 'exec'), namespace)
        secret = b'b' * 64
        with patch.object(namespace['subprocess'], 'run', return_value=subprocess.CompletedProcess([], 0)) as runner:
            self.assertEqual(namespace['store_credential'](secret), 0)
        self.assertEqual(runner.call_args.args[0], ['/usr/bin/security', '-i'])
        self.assertIn(secret, runner.call_args.kwargs['input'])
        self.assertTrue(runner.call_args.kwargs['input'].endswith(b'\n'))
        self.assertEqual(runner.call_args.kwargs['stdout'], subprocess.DEVNULL)
        self.assertEqual(runner.call_args.kwargs['stderr'], subprocess.DEVNULL)
        self.assertEqual(runner.call_args.kwargs['timeout'], 20)

    def test_gui_worker_fifo_status_has_no_credential(self):
        namespace = {'__name__': 'keychain_fixture'}
        exec(compile(share.guest_keychain_worker(), '<fixture-worker>', 'exec'), namespace)
        secret = b'c' * 64
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            fifo, status = Path(temp) / 'input', Path(temp) / 'status.json'
            os.mkfifo(fifo, 0o600)
            def writer():
                with fifo.open('wb') as stream:
                    stream.write(secret)
            thread = threading.Thread(target=writer)
            with patch.object(namespace['sys'], 'argv', ['worker', str(fifo), str(status)]), patch.object(namespace['signal'], 'signal'), patch.object(namespace['signal'], 'alarm'), patch.object(namespace['subprocess'], 'run', return_value=subprocess.CompletedProcess([], 0)):
                thread.start()
                self.assertEqual(namespace['main'](), 0)
                thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
            self.assertEqual(json.loads(status.read_text()), {'returncode': 0})
            self.assertNotIn(secret.decode(), status.read_text())
            self.assertEqual(fifo.stat().st_mode & 0o777, 0o600)

    def test_gui_worker_invalid_fifo_data_returns_secretless_failure(self):
        namespace = {'__name__': 'keychain_fixture'}
        exec(compile(share.guest_keychain_worker(), '<fixture-worker>', 'exec'), namespace)
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            fifo, status = Path(temp) / 'input', Path(temp) / 'status.json'
            os.mkfifo(fifo, 0o600)
            def writer():
                with fifo.open('wb') as stream:
                    stream.write(b'$(bad input)')
            thread = threading.Thread(target=writer)
            with patch.object(namespace['sys'], 'argv', ['worker', str(fifo), str(status)]), patch.object(namespace['signal'], 'signal'), patch.object(namespace['signal'], 'alarm'), patch.object(namespace['subprocess'], 'run') as runner:
                thread.start()
                self.assertEqual(namespace['main'](), 1)
                thread.join(timeout=2)
                runner.assert_not_called()
            self.assertEqual(json.loads(status.read_text()), {'returncode': 1, 'reason': 'ValueError'})
            self.assertNotIn('bad input', status.read_text())

    def test_credentials_reject_shell_input(self):
        for password in ('$(id)', 'a' * 63, 'a' * 64 + '\n', '"' * 64):
            with self.assertRaises(ValueError):
                share.guest_installer(password)

    def test_stop_does_not_touch_foreign_container(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp, patch.object(share.os, 'getuid', return_value=1000):
            service = share.HomeShare(temp, home=temp)
            service.container = 'fixture-id'
            def docker(*args, **_):
                return subprocess.CompletedProcess(args, 0, '[{"Config":{"Labels":{}}}]', '')
            with patch.object(service, 'docker', side_effect=docker) as mocked:
                with self.assertRaises(RuntimeError):
                    service.stop()
                self.assertEqual(mocked.call_count, 1)

    def test_start_keeps_database_in_tmpfs_and_no_password_argv(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp, patch.object(share.os, 'getuid', return_value=1000):
            root = Path(temp)
            service = share.HomeShare(root, home=root, state=root / 'config', port=0)
            calls = []
            def docker(*args, **_):
                calls.append(args)
                if args[0] == 'run':
                    return subprocess.CompletedProcess(args, 0, 'fixture-id', '')
                if args[0] == 'inspect' and '--format' in args:
                    return subprocess.CompletedProcess(args, 0, 'true', '')
                return subprocess.CompletedProcess(args, 1, '', '')
            with patch.object(service, 'docker', side_effect=docker), patch.object(share.socket, 'create_connection'):
                service.start()
            command = next(c for c in calls if c[0] == 'run')
            self.assertIn('host', command)
            self.assertIn('/var/lib/samba', command)
            self.assertFalse(any('dst=/var/lib/samba/private' in c for c in command))
            credential = (root / 'config/password').read_text().strip()
            self.assertFalse(any(credential in c for c in command))
            self.assertEqual((root / 'config/password').stat().st_mode & 0o777, 0o600)

    def test_bootstrap_stops_without_ssh(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp, patch.object(share.os, 'getuid', return_value=1000):
            service = share.HomeShare(temp, home=temp)
            service.stop_event.set()
            with patch.object(share.subprocess, 'run') as mocked:
                result = service.bootstrap(['ssh', 'fixture'])
                self.assertFalse(result['installed'])
                self.assertTrue(result['stopped'])
                mocked.assert_not_called()


if __name__ == '__main__':
    unittest.main()
