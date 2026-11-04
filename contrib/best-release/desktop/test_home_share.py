#!/usr/bin/env python3
"""No VM, no Samba, no real home, and no live credentials in these tests."""
import importlib.util
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

    def test_guest_credential_only_stdin_not_persistent_files(self):
        secret = 'a' * 64
        payload = share.guest_installer(secret)
        self.assertEqual(payload.count(secret), 1)
        self.assertIn('security -i >/dev/null 2>&1 <<KEYCHAIN', payload)
        self.assertIn('login.keychain-db', payload)
        agent = share.guest_agent()
        self.assertNotIn(secret, agent)
        self.assertIn('find-internet-password', agent)
        self.assertIn('with password sharePassword', agent)
        self.assertIn('StartInterval', str(__import__('plistlib').loads(__import__('base64').b64decode(payload.split('printf %s ')[2].split(' |')[0]))))
        shell = subprocess.run(['/bin/sh', '-n'], input=payload, text=True, capture_output=True)
        self.assertEqual(shell.returncode, 0, shell.stderr)
        self.assertEqual(subprocess.run(['/bin/sh', '-n'], input=agent, text=True).returncode, 0)

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
