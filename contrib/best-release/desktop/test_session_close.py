"""Mock the actual supervisor: no VM, SSH, service, or process control executed."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch, Mock

P = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('session', P / 'run-session.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='', dir=P)
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name)
        self.args = ['/fixture/qemu', '-drive', 'id=MacHDD,if=none,file=' + str(self.run / 'macos.qcow2'),
                     '-netdev', 'user,id=net0,hostfwd=tcp:127.0.0.1:11041-:22,hostfwd=tcp:127.0.0.1:5901-:5900',
                     '-qmp', 'unix:' + str(self.run / 'qmp.sock') + ',server=on,wait=off',
                     '-qmp', 'unix:' + str(self.run / 'close-events.sock') + ',server=on,wait=off']
        (self.run / 'build-binary.json').write_text('{"channel":"best"}')
        (self.run / 'command.json').write_text(json.dumps(self.args))

    def test_derived_helper_args_and_opt_in_boundary(self):
        command = s.close_helper_args(self.run, self.args)
        self.assertEqual(command[command.index('--ssh-port') + 1], '11041')
        self.assertEqual(command[command.index('--disk') + 1], str(self.run / 'macos.qcow2'))
        self.assertEqual(command[command.index('--executable') + 1], '/fixture/qemu')
        self.assertEqual(command[command.index('--ssh-key') + 1], str(Path.home() / 'macos-vm/vmkey'))
        self.assertIsNone(s.close_helper_args(self.run, self.args[:-2]))
        (self.run / 'build-binary.json').write_text('{"channel":"daily"}')
        self.assertIsNone(s.close_helper_args(self.run, self.args))

    def test_rejects_missing_or_ambiguous_forward_before_vm(self):
        no_forward = [arg.replace('hostfwd=tcp:127.0.0.1:11041-:22,', '') for arg in self.args]
        with self.assertRaisesRegex(RuntimeError, 'one localhost SSH'):
            s.close_helper_args(self.run, no_forward)
        with self.assertRaisesRegex(RuntimeError, 'one localhost SSH'):
            s.close_helper_args(self.run, self.args + ['-netdev', 'user,id=net2,hostfwd=tcp:127.0.0.1:11042-:22'])

    def test_real_main_spawns_helper_with_child_pid_and_cleans_only_owned_helpers(self):
        qemu = Mock(pid=1234)
        qemu.wait.return_value = 0
        qemu.poll.return_value = 0
        helper, controls = Mock(pid=1235), Mock(pid=1236)
        helper.poll.return_value = None
        controls.poll.return_value = None
        lease = Mock()
        original_open = open
        def opened(file, *args, **kwargs):
            if str(file) == '/var/tmp/reims-host-workload.lock':
                return lease
            return original_open(file, *args, **kwargs)
        with patch.object(s.sys, 'argv', ['run-session', str(self.run)]), patch.dict(s.os.environ, {}, clear=True), patch('builtins.open', side_effect=opened), patch.object(s.fcntl, 'flock'), patch.object(s.os, 'umask'), patch.object(s.signal, 'signal'), patch.object(s.subprocess, 'Popen', side_effect=[qemu, helper, controls]) as popen, patch.object(s.subprocess, 'run') as calls:
            self.assertEqual(s.main(), 0)
        helper_command = popen.call_args_list[1].args[0]
        self.assertEqual(helper_command[helper_command.index('--pid') + 1], '1234')
        self.assertTrue(helper_command[2].endswith('/close-guest.py'))
        qemu.terminate.assert_not_called()
        qemu.kill.assert_not_called()
        helper.terminate.assert_called_once()
        helper.wait.assert_called_once_with(timeout=12)
        controls.terminate.assert_called_once()
        self.assertEqual(len(calls.call_args_list), 2)  # existing owned USB cleanup, fully mocked
        self.assertEqual(json.loads((self.run / 'exit.json').read_text()), {'qemu_exit_code':0})

    def test_stubborn_helper_cleanup_is_bounded(self):
        helper = Mock()
        helper.poll.return_value = None
        helper.wait.side_effect = [subprocess.TimeoutExpired('helper', 12), 0]
        s.stop_close_helper(helper)
        helper.terminate.assert_called_once()
        helper.kill.assert_called_once()
        self.assertEqual([call.kwargs['timeout'] for call in helper.wait.call_args_list], [12, 2])


if __name__ == '__main__':
    unittest.main(verbosity=2)
