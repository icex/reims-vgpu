import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest import mock

BASE = Path(__file__).resolve().parent
PACKAGE = BASE / 'guest-clipboard'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


installer = load('installer', PACKAGE / 'install.py')
clip = load('guest_clipboard', PACKAGE / 'console-vdagent-clipboard.py')
agent = load('guest_agent', PACKAGE / 'console-vdagent-agent.py')
host = load('host_installer', BASE / 'guest-clipboard.py')


class Installation(unittest.TestCase):
    def test_bootstrap_is_idempotent_and_preserves_legacy_state(self):
        with tempfile.TemporaryDirectory(dir=BASE) as temporary:
            home = Path(temporary)
            agents = home / 'Library/LaunchAgents'
            agents.mkdir(parents=True)
            legacy = agents / 'org.raphaelgpu.clipboard.plist'
            legacy.write_bytes(b'legacy plist stays identical')
            prior = agents / 'org.reims.clipboard.plist'
            prior.write_bytes(b'previous reims plist')
            loaded = {installer.LEGACY, installer.LABEL}
            disabled = {installer.LEGACY: False, installer.LABEL: False}
            calls = []
            def call(argv, check=False, timeout=15):
                calls.append(argv)
                output = b''
                code = 0
                if argv[0] == 'xcrun':
                    Path(argv[-1]).write_bytes(b'fake helper executable')
                elif argv[0].endswith('console-clipboard'):
                    self.assertEqual(argv[1:], ['read', '-1'])
                    output = struct.pack('<qI', 10, 0)
                elif argv[:2] == ['launchctl', 'print'] and '/' in argv[2][4:]:
                    code = 0 if argv[2].split('/')[-1] in loaded else 1
                elif argv[:2] == ['launchctl', 'print-disabled']:
                    output = ('\n'.join('"' + k + '" => ' + str(v).lower() for k, v in disabled.items())).encode()
                elif argv[:2] == ['launchctl', 'bootout']:
                    loaded.discard(argv[2].split('/')[-1])
                elif argv[:2] == ['launchctl', 'bootstrap']:
                    loaded.add(Path(argv[3]).stem)
                elif argv[:2] in (['launchctl', 'enable'], ['launchctl', 'disable']):
                    disabled[argv[2].split('/')[-1]] = argv[1] == 'disable'
                return subprocess.CompletedProcess(argv, code, output, b'')
            with mock.patch.object(installer.Path, 'home', return_value=home), mock.patch.object(installer, 'call', side_effect=call), mock.patch('sys.stdout', new_callable=io.StringIO) as stdout:
                installer.install()
                first = json.loads(stdout.getvalue())
                before = len(calls)
                stdout.seek(0);stdout.truncate()
                installer.install()
                second = json.loads(stdout.getvalue())
            self.assertTrue(first['restarted'])
            self.assertTrue(first['compiled'])
            self.assertFalse(second['restarted'])
            self.assertFalse(second['compiled'])
            self.assertFalse(any(a[0] == 'xcrun' or a[:2] == ['launchctl', 'bootstrap'] for a in calls[before:]))
            self.assertEqual(legacy.read_bytes(), b'legacy plist stays identical')
            root = home / 'Library/Application Support/Reims/clipboard-agent'
            migration = json.loads((root / 'migration.json').read_text())
            self.assertTrue(migration['legacy_loaded'])
            self.assertFalse(migration['legacy_disabled'])
            self.assertEqual((root / 'previous-org.reims.clipboard.plist').read_bytes(), b'previous reims plist')
            self.assertNotIn(installer.LEGACY, loaded)
            self.assertTrue(disabled[installer.LEGACY])
            self.assertIn(installer.LABEL, loaded)
            self.assertEqual((root / 'migration.json').stat().st_mode & 0o777, 0o600)
            for name in installer.FILES:
                self.assertEqual((root / name).read_bytes(), (PACKAGE / name).read_bytes())

    def test_sources_have_no_binary_and_fixed_remote_command(self):
        self.assertEqual(set(host.FILES), {'install.py', *installer.FILES})
        for name in host.FILES:
            (PACKAGE / name).read_text()
        argv = host.ssh(10023, Path('/key with spaces'), Path('/known hosts'))
        self.assertIn('/key with spaces', argv)
        self.assertEqual(argv[-1], 'bogdan@127.0.0.1')
        self.assertIn('StrictHostKeyChecking=yes', argv)


class TransientBackend(unittest.TestCase):
    def test_helper_failure_recovers_and_success_clears_attempts(self):
        backend = clip.Pasteboard('/fake helper')
        with mock.patch.object(clip, 'helper_call', side_effect=[subprocess.TimeoutExpired('helper', 3), (-14, b''), (0, struct.pack('<qI', 11, 0))]):
            for number in (1, 2):
                with self.assertRaises(clip.BackendUnavailable) as failure:
                    backend.read(-1)
                self.assertEqual(failure.exception.attempt, number)
            self.assertEqual(backend.read(-1), (11, 0, None))
            self.assertEqual(backend.failures, 0)

    def test_recovery_invalidates_mutation_but_drains_stale_response(self):
        class Backend:
            def read(self, expected): return 17, 0, None
        bridge = clip.Clipboard(Backend(), lambda _: None, clock=lambda: 5)
        bridge.ready = True
        bridge.count = 10
        bridge.generation = 3
        bridge.inflight = (3, 10, 4)
        bridge.want = True
        bridge.owned = True
        bridge.backend_unavailable()
        self.assertFalse(bridge.disabled)
        self.assertEqual(bridge.inflight, (3, 10, 4))
        self.assertFalse(bridge.want)
        self.assertIsNone(bridge.count)
        self.assertEqual(bridge.next_poll, 6)
        row = {'port': 1, 'type': 4, 'clipboard_payload': struct.pack('<I', 1) + b'late'}
        self.assertEqual(bridge.handle(row, 10), b'')
        self.assertIsNone(bridge.inflight)
        self.assertFalse(bridge.owned)

    def test_actual_agent_recovery_branch_replies_unavailable(self):
        class FakeClipboard:
            def __init__(self, *args):
                self.ready = True
                self.disabled = False
                self.recovered = False
            def poll(self, remaining): return b''
            def handle(self, *args): raise agent.clipboard_module.BackendUnavailable(1)
            def backend_unavailable(self): self.recovered = True
        class Parser:
            def __init__(self, **kwargs): self.validator = self
            def feed(self, data): return [{'port': 1, 'type': 8, 'clipboard_payload': struct.pack('<I', 1)}]
            def pending(self): return {'per_port_message_bytes': {'1': 0, '2': 0}}
        now = [0]
        def clock():
            now[0] += .01
            return now[0]
        events = []
        sent = []
        reads = [True]
        def select(read, write, error, timeout):
            incoming = reads.pop() if reads else False
            return ([42] if incoming else [], write, [])
        with mock.patch.object(agent.clipboard_module, 'Clipboard', FakeClipboard), mock.patch.object(agent.monitors, 'MonitorParser', Parser), mock.patch.object(agent.select, 'select', side_effect=select), mock.patch.object(agent.os, 'read', return_value=b'fake'), mock.patch.object(agent.os, 'write', side_effect=lambda _, data: sent.append(bytes(data)) or len(data)):
            result = agent.serve(42, .5, lambda *args: {}, events.append, clock, clipboard_backend=object())
        self.assertIn({'event': 'clipboard-backend-retry', 'attempt': 1}, events)
        wire = b''.join(sent)
        self.assertIn(agent.clipboard_module.packet(4, struct.pack('<I', 0)), wire)
        self.assertNotIn('clipboard-disabled', json.dumps(events))
        self.assertEqual(result['reason'], 'session-deadline')


if __name__ == '__main__': unittest.main()
