"""No VM/GPU/SSH execution: owned Unix event peer and mocked SSH children."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch, Mock

P = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('helper', P / 'close-events.py')
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='', dir=os.environ.get('TMPDIR', P))
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root / 'r'
        self.run.mkdir()
        self.proc = self.root / 'proc'
        self.pid = os.getpid()
        self.exe = self.root / 'test-executable'
        self.exe.write_text('not executable; identity fixture only')
        self.procdir = self.proc / str(self.pid)
        self.procdir.mkdir(parents=True)
        (self.procdir / 'exe').symlink_to(self.exe)
        self.disk = self.run / 'macos.qcow2'
        self.cmdline(self.disk)
        self.records = patch.object(h, 'record').start()
        self.addCleanup(patch.stopall)

    def cmdline(self, disk):
        (self.procdir / 'cmdline').write_bytes(str(self.exe).encode() + b'\0-drive\0if=none,format=qcow2,file=' + str(disk).encode() + b'\0')

    def test_peer_identity_pid_exe_and_exact_disk(self):
        good = (self.pid, os.getuid(), 123)
        h.validate_identity(self.run, self.pid, self.exe, good, self.proc)
        for peer in [(self.pid + 1, os.getuid(), 123), (self.pid, os.getuid() + 1, 123)]:
            with self.subTest(peer=peer), self.assertRaisesRegex(RuntimeError, 'pid or owner'):
                h.validate_identity(self.run, self.pid, self.exe, peer, self.proc)
        other = self.root / 'other'
        other.write_text('wrong executable')
        with self.assertRaisesRegex(RuntimeError, 'executable'):
            h.validate_identity(self.run, self.pid, other, good, self.proc)
        self.cmdline(Path(str(self.disk) + '.spoof'))
        with self.assertRaisesRegex(RuntimeError, 'run disk'):
            h.validate_identity(self.run, self.pid, self.exe, good, self.proc)
        alternate = self.root / 'isolated-disk' / 'macos.qcow2'
        self.cmdline(alternate)
        h.validate_identity(self.run, self.pid, self.exe, good, self.proc, alternate)

    def test_dedicated_socket_negotiation_and_interleaved_events(self):
        endpoint = self.run / 'close-events.sock'
        errors, requests = [], []
        listener = socket.socket(socket.AF_UNIX)
        listener.bind(str(endpoint))
        listener.listen(1)
        def serve():
            try:
                with listener.accept()[0] as conn, conn.makefile('rb') as channel:
                    conn.sendall(b'{"QMP":{"version":{}}}\n')
                    requests.append(json.loads(channel.readline()))
                    data = b'{"event":"POWERDOWN"}\n{"return":{},"id":"close-events"}\n{"event":"RTC_CHANGE"}\n'
                    conn.sendall(data[:7])
                    conn.sendall(data[7:])
            except BaseException as error:
                errors.append(error)
            finally:
                listener.close()
        thread = threading.Thread(target=serve)
        thread.start()
        events = h.QMPEvents(self.run, self.pid, self.exe, lambda:False, self.proc)
        try:
            self.assertEqual(events.next()['event'], 'POWERDOWN')
            self.assertEqual(events.next()['event'], 'RTC_CHANGE')
            with self.assertRaises(EOFError):
                events.next()
        finally:
            events.close()
            thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(requests, [{'execute':'qmp_capabilities', 'id':'close-events'}])
        self.assertFalse((self.run / 'qmp.sock').exists())

    def test_irrelevant_and_repeated_events_launch_once_per_pending_period(self):
        clock = [0.0]
        child = Mock(pid=456, returncode=0)
        child.poll.return_value = 0
        child.communicate.return_value = ('request accepted', '')
        with patch.object(h.time, 'monotonic', side_effect=lambda:clock[0]), patch.object(h, 'notify'), patch.object(h.subprocess, 'Popen', return_value=child) as popen:
            handler = h.PowerdownHandler(12345, self.root / 'key', self.run)
            handler.event({'event':'RTC_CHANGE'})
            popen.assert_not_called()
            handler.event({'event':'POWERDOWN'})
            handler.event({'event':'POWERDOWN'})
            handler.event({'event':'POWERDOWN'})
            popen.assert_called_once()
            argv = popen.call_args.args[0]
            self.assertEqual(argv[0], 'ssh')
            self.assertIn('UserKnownHostsFile=' + str(self.run / 'close-known_hosts'), argv)
            self.assertEqual(argv[argv.index('-p') + 1], '12345')
            self.assertEqual(argv[-1], h.GUEST_SHUTDOWN)
            self.assertNotIn('saving no', argv[-1])
            self.assertNotIn('kill', argv[-1])
            clock[0] = h.PENDING_SECONDS + 1
            handler.event({'event':'POWERDOWN'})
            self.assertEqual(popen.call_count, 2)
            child.terminate.assert_not_called()
            child.kill.assert_not_called()
            handler.close()

    def test_ssh_failure_or_launch_error_never_controls_qemu(self):
        child = Mock(pid=456, returncode=255)
        child.poll.return_value = 255
        child.communicate.return_value = ('', 'connection refused')
        with patch.object(h, 'notify') as notify, patch.object(h.subprocess, 'Popen', return_value=child) as popen:
            handler = h.PowerdownHandler(12345, self.root / 'key', self.run)
            handler.event({'event':'POWERDOWN'})
            handler.tick()
            handler.event({'event':'POWERDOWN'})
            popen.assert_called_once()
            self.assertIn('retry automatically', notify.call_args.args[0])
            self.assertIsNotNone(handler.retry_at)
            child.terminate.assert_not_called()
            child.kill.assert_not_called()
        with patch.object(h, 'notify') as notify, patch.object(h.subprocess, 'Popen', side_effect=OSError('no ssh')) as popen:
            handler = h.PowerdownHandler(12345, self.root / 'key', self.run)
            handler.event({'event':'POWERDOWN'})
            handler.event({'event':'POWERDOWN'})
            popen.assert_called_once()
            self.assertIn('VM remains running', notify.call_args.args[0])

    def test_timeout_stops_only_owned_ssh_transport(self):
        clock = [0.0]
        child = Mock(pid=456, returncode=-15)
        child.poll.side_effect = [None, None, -15]
        child.communicate.return_value = ('', 'transport stopped')
        with patch.object(h.time, 'monotonic', side_effect=lambda:clock[0]), patch.object(h, 'notify'), patch.object(h.subprocess, 'Popen', return_value=child) as popen:
            handler = h.PowerdownHandler(12345, self.root / 'key', self.run)
            handler.event({'event':'POWERDOWN'})
            clock[0] = h.SSH_TIMEOUT + 1
            handler.tick()
            child.terminate.assert_called_once()
            child.wait.assert_called_once_with(timeout=2)
            self.assertEqual(popen.call_args.args[0][0], 'ssh')


if __name__ == '__main__':
    unittest.main(verbosity=2)
