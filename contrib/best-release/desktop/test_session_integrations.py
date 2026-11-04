#!/usr/bin/env python3
"""Exercise the actual session supervisor with fake processes and isolated files.

REIMS_SESSION_TEST_TARGET can point at a staged integration supervisor. No QEMU,
SSH, Docker, service operation, real lease, or process signal is executed.
"""
from contextlib import ExitStack
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
TARGET = Path(os.environ.get('REIMS_SESSION_TEST_TARGET', HERE / 'run-session.py'))
spec = importlib.util.spec_from_file_location('integrated_session', TARGET)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


class SessionIntegrations(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='session-fixture-', dir=HERE)
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.run = self.directory / 'run'
        self.run.mkdir()
        (self.directory / 'close-guest.py').touch()
        self.command = ['/fixture/qemu', '-drive', 'file=' + str(self.run / 'macos.qcow2'),
                        '-netdev', 'user,id=net0,hostfwd=tcp:127.0.0.1:11041-:22',
                        '-qmp', 'unix:' + str(self.run / 'close-events.sock') + ',server=on,wait=off']
        (self.run / 'command.json').write_text(json.dumps(self.command))
        (self.run / 'build-binary.json').write_text('{"channel":"best"}')
        self.events = []
        self.children = {}
        for name in ('qemu', 'home-share', 'guest-clipboard', 'file-drop', 'close-guest', 'controls'):
            child = Mock(name=name, pid=2000 + len(self.children))
            child.poll.return_value = 0 if name == 'qemu' else None
            child.wait.return_value = 0
            self.children[name] = child
        self.children['qemu'].wait.side_effect = lambda **_: self.events.append('wait:qemu') or 0
        self.fail_spawn = set()
        self.exited = set()
        self.on_spawn = None

    def invoke(self, settings, *, extra_env=None, stop_optional=None):
        (self.directory / 'desktop-settings.json').write_text(json.dumps(settings))
        original_open = open
        lease = Mock()
        def opened(path, *args, **kwargs):
            if str(path) == '/var/tmp/reims-host-workload.lock':
                return lease
            return original_open(path, *args, **kwargs)
        def popen(argv, **_):
            if argv[0] == '/fixture/qemu':
                name = 'qemu'
                self.qemu_environment = dict(s.os.environ)
            else:
                file = next(Path(a).name for a in argv if a.endswith('.py'))
                name = {'desktop-controls.py': 'controls'}.get(file, file.removesuffix('.py'))
            self.events.append('spawn:' + name)
            if self.on_spawn is not None:
                self.on_spawn(name)
            if name in self.fail_spawn:
                raise OSError('injected spawn failure')
            child = self.children[name]
            if name in self.exited:
                child.poll.return_value = 1
            return child
        def register_signal(number, handler):
            self.events.append('signal:' + str(number))
            self.handlers[number] = handler
        self.handlers = {}
        with ExitStack() as stack:
            stack.enter_context(patch.object(s, '__file__', str(self.directory / 'run-session.py')))
            stack.enter_context(patch.object(s.sys, 'argv', ['run-session', str(self.run)]))
            stack.enter_context(patch.dict(s.os.environ, extra_env or {}, clear=True))
            stack.enter_context(patch('builtins.open', side_effect=opened))
            stack.enter_context(patch.object(s.fcntl, 'flock'))
            stack.enter_context(patch.object(s.os, 'umask'))
            stack.enter_context(patch.object(s.signal, 'signal', side_effect=register_signal))
            stack.enter_context(patch.object(s.subprocess, 'Popen', side_effect=popen))
            self.subprocess_run = stack.enter_context(patch.object(s.subprocess, 'run'))
            stack.enter_context(patch.object(s.Path, 'is_socket', return_value=True))
            if stop_optional is not None:
                stack.enter_context(patch.object(s, 'stop_optional', side_effect=stop_optional))
            return s.main()

    def test_enabled_services_start_before_qemu_and_cleanup(self):
        self.assertEqual(self.invoke({'home_share': True, 'clipboard': True, 'file_drop': True}), 0)
        qemu_at = self.events.index('spawn:qemu')
        for name in ('home-share', 'guest-clipboard', 'file-drop'):
            self.assertLess(self.events.index('spawn:' + name), qemu_at)
            self.children[name].terminate.assert_called_once()
            self.children[name].wait.assert_called_once()
        self.assertEqual(len([e for e in self.events[:2] if e.startswith('signal:')]), 2)
        self.assertEqual(self.qemu_environment['REIMS_VGPU_DROP_SOCKET'], str(self.run / 'drop.sock'))
        self.assertEqual(self.qemu_environment['REIMS_CLIPBOARD'], '1')
        self.children['qemu'].terminate.assert_not_called()
        self.children['qemu'].kill.assert_not_called()
        self.assertEqual(json.loads((self.run / 'exit.json').read_text()), {'qemu_exit_code': 0})
        self.assertEqual(self.subprocess_run.call_count, 2)  # owned USB cleanup only

    def test_absent_feature_keys_preserve_previous_session_behaviour(self):
        self.assertEqual(self.invoke({}, extra_env={'REIMS_VGPU_DROP_SOCKET': '/untrusted/stale'}), 0)
        for name in ('home-share', 'guest-clipboard', 'file-drop'):
            self.assertNotIn('spawn:' + name, self.events)
        self.assertNotIn('REIMS_VGPU_DROP_SOCKET', self.qemu_environment)
        self.assertEqual(self.qemu_environment['REIMS_CLIPBOARD'], '1')

    def test_independent_switches_and_controls_failure_keep_vm_running(self):
        self.fail_spawn = {'home-share', 'controls'}
        self.assertEqual(self.invoke({'home_share': True, 'clipboard': False, 'file_drop': True}), 0)
        self.assertIn('spawn:qemu', self.events)
        self.assertIn('wait:qemu', self.events)
        self.assertNotIn('spawn:guest-clipboard', self.events)
        self.assertEqual(self.qemu_environment['REIMS_CLIPBOARD'], '0')
        self.children['file-drop'].terminate.assert_called_once()
        self.children['qemu'].terminate.assert_not_called()

    def test_optional_workers_that_exit_with_failure_never_block_vm(self):
        self.exited = {'home-share', 'guest-clipboard', 'file-drop'}
        self.assertEqual(self.invoke({'home_share': True, 'clipboard': True, 'file_drop': True}), 0)
        self.assertIn('wait:qemu', self.events)
        self.assertNotIn('REIMS_VGPU_DROP_SOCKET', self.qemu_environment)
        for name in self.exited:
            self.children[name].terminate.assert_not_called()

    def test_qemu_spawn_failure_cleans_all_started_services(self):
        self.fail_spawn = {'qemu'}
        with self.assertRaises(OSError):
            self.invoke({'home_share': True, 'clipboard': True, 'file_drop': True})
        for name in ('home-share', 'guest-clipboard', 'file-drop'):
            self.children[name].terminate.assert_called_once()
        self.assertNotIn('spawn:close-guest', self.events)
        self.assertNotIn('spawn:controls', self.events)
        self.assertFalse((self.run / 'exit.json').exists())

    def test_one_cleanup_failure_does_not_skip_home_share(self):
        original = s.stop_optional
        def cleanup(child):
            if child is self.children['file-drop']:
                raise RuntimeError('injected drop cleanup failure')
            return original(child)
        self.assertEqual(self.invoke({'home_share': True, 'clipboard': True, 'file_drop': True}, stop_optional=cleanup), 0)
        self.children['home-share'].terminate.assert_called_once()
        self.children['guest-clipboard'].terminate.assert_called_once()
        self.children['close-guest'].terminate.assert_called_once()
        self.children['controls'].terminate.assert_called_once()
        self.assertEqual(self.subprocess_run.call_count, 2)

    def test_shutdown_signal_during_startup_never_launches_qemu(self):
        def request_stop(name):
            if name == 'file-drop':
                self.handlers[s.signal.SIGTERM](s.signal.SIGTERM, None)
        self.on_spawn = request_stop
        result = self.invoke({'home_share': True, 'clipboard': True, 'file_drop': True})
        self.assertIn(result, (0, 128 + int(s.signal.SIGTERM)))
        self.assertNotIn('spawn:qemu', self.events)
        for name in ('home-share', 'guest-clipboard', 'file-drop'):
            self.children[name].terminate.assert_called_once()


if __name__ == '__main__':
    unittest.main(verbosity=2)
