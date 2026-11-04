import ctypes
import importlib.util
from pathlib import Path
import subprocess
import sys
import threading
import time
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location('clip', Path(__file__).with_name('wayland-clipboard.py'))
clip = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clip)


class Channel:
    def __init__(self): self.calls = []
    def clipboard_selection_grab(self, *args): self.calls.append(('grab', args))
    def clipboard_selection_release(self, *args): self.calls.append(('release', args))
    def clipboard_selection_notify(self, *args): self.calls.append(('notify', args))
    def clipboard_selection_request(self, *args): self.calls.append(('request', args))


class Protocol(unittest.TestCase):
    def setUp(self):
        self.channel = Channel()
        self.writes = []
        self.now = 0
        self.bridge = clip.Clipboard(self.channel, lambda *args: self.writes.append(args), clock=lambda: self.now)
        self.bridge.set_enabled(True)
        self.bridge.set_connected(True)
        self.bridge.host_changed(b'baseline')

    def offer(self, data):
        self.bridge.grabbed(None, 0, [1], 1)
        self.bridge.received(None, 0, 1, data, len(data))

    def test_baseline_is_not_shared(self):
        self.assertEqual(self.channel.calls, [])

    def test_host_sizes_unicode_empty_and_no_echo(self):
        for value in [b'', b'x', 'naïve 日本語 😀\n'.encode(), b'a' * clip.LIMIT]:
            self.bridge.host_changed(value)
            before = len(self.channel.calls)
            self.bridge.host_changed(value)
            self.bridge.requested(None, 0, 1)
            self.assertEqual(len(self.channel.calls), before + 1)
            self.assertEqual(self.channel.calls[-1], ('notify', (0, 1, list(value))))

    def test_guest_write_not_reoffered_or_repeated(self):
        data = 'guest café'.encode()
        self.offer(data)
        generation = self.writes[-1][2]
        self.bridge.applied(generation, True, data)
        before = len(self.channel.calls)
        for _ in range(10): self.bridge.host_changed(data)
        self.assertEqual(len(self.channel.calls), before)
        self.assertEqual(self.writes, [(b'baseline', data, generation)])

    def test_stale_guest_response_never_overwrites_host(self):
        self.bridge.grabbed(None, 0, [1], 1)
        self.bridge.host_changed(b'newer host')
        self.bridge.received(None, 0, 1, b'old guest', 9)
        self.assertEqual(self.writes, [])
        self.assertEqual(self.bridge.host, b'newer host')

    def test_new_guest_offer_waits_for_old_response(self):
        self.bridge.grabbed(None, 0, [1], 1)
        self.bridge.grabbed(None, 0, [1], 1)
        self.assertEqual(sum(x[0] == 'request' for x in self.channel.calls), 1)
        self.bridge.received(None, 0, 1, b'old', 3)
        self.assertEqual(self.writes, [])
        self.assertEqual(sum(x[0] == 'request' for x in self.channel.calls), 2)
        self.bridge.received(None, 0, 1, b'new', 3)
        self.assertEqual(self.writes[-1][1], b'new')

    def test_toggle_blocks_pending_and_new_transfers(self):
        self.bridge.grabbed(None, 0, [1], 1)
        self.bridge.set_enabled(False)
        self.bridge.received(None, 0, 1, b'pending', 7)
        self.bridge.host_changed(b'private')
        self.bridge.grabbed(None, 0, [1], 1)
        self.assertEqual(self.writes, [])
        self.bridge.set_enabled(True)
        self.bridge.host_changed(b'private')
        self.assertFalse(self.bridge.owned)

    def test_selection_release_does_not_clear_clipboard(self):
        self.bridge.released(None, 0)
        self.assertEqual(self.bridge.host, b'baseline')
        self.assertEqual(self.writes, [])

    def test_empty_is_transferred_and_absent_is_not_empty(self):
        self.offer(b'')
        self.assertEqual(self.writes[-1][1], b'')
        self.bridge.applied(self.writes[-1][2], True, b'')
        self.bridge.host_changed(None)
        self.assertEqual(self.channel.calls[-1][0], 'release')

    def test_unrequested_wrong_selection_and_malformed_data_refused(self):
        self.bridge.received(None, 0, 1, b'unsolicited', 11)
        self.bridge.grabbed(None, 1, [1], 1)
        self.assertEqual(self.channel.calls, [])
        for value in [b'\0', b'\xff', b'a' * (clip.LIMIT + 1)]:
            self.offer(value)
        self.assertEqual(self.writes, [])

    def test_ctypes_gobject_pointer_shapes(self):
        types = (ctypes.c_uint32 * 2)(4, 1)
        self.bridge.grabbed(None, 0, ctypes.addressof(types), 2)
        data = ctypes.create_string_buffer('日本語'.encode())
        self.bridge.received(None, 0, 1, ctypes.addressof(data), len(data.raw) - 1)
        self.assertEqual(self.writes[-1][1], '日本語'.encode())

    def test_timeout_releases_once_and_never_starts_ambiguous_request(self):
        self.bridge.grabbed(None, 0, [1], 1)
        self.now = 6
        for _ in range(8): self.bridge.tick()
        self.assertTrue(self.bridge.disabled)
        self.assertEqual(sum(x[0] == 'release' for x in self.channel.calls), 1)
        self.bridge.grabbed(None, 0, [1], 1)
        self.assertEqual(sum(x[0] == 'request' for x in self.channel.calls), 1)

    def test_guest_offer_before_host_baseline_is_delayed(self):
        self.bridge.set_connected(False)
        self.bridge.set_connected(True)
        self.bridge.grabbed(None, 0, [1], 1)
        self.assertFalse(self.channel.calls)
        self.bridge.host_changed(b'current')
        self.assertEqual(self.channel.calls[-1][0], 'request')


class Backend(unittest.TestCase):
    def test_bounded_provider_success_and_oversize(self):
        self.assertEqual(clip.bounded_read([sys.executable, '-c', "import sys;sys.stdout.buffer.write(b'x'*65536)"]), b'x' * 65536)
        with self.assertRaises(ValueError):
            clip.bounded_read([sys.executable, '-c', "import sys;sys.stdout.buffer.write(b'x'*1000000)"])
        with self.assertRaises(TimeoutError):
            clip.bounded_read([sys.executable, '-c', 'import time;time.sleep(2)'], .03)

    def test_payload_stays_in_stdin(self):
        value = b'$(touch /should-never-run)\n\'"\\'
        with mock.patch.object(subprocess, 'run') as run:
            clip.WaylandClipboard().write(value)
            self.assertEqual(run.call_args.args[0], ['wl-copy', '--type', 'text/plain;charset=utf-8'])
            self.assertEqual(run.call_args.kwargs['input'], value)
            self.assertNotIn('shell', run.call_args.kwargs)

    def test_worker_newer_host_wins(self):
        class Fake:
            def __init__(self): self.writes = []
            def read(self): return b'newer host'
            def write(self, data): self.writes.append(data)
        fake = Fake()
        applied = []
        event = threading.Event()
        def done(*args): applied.append(args);event.set()
        worker = clip.Worker(fake, lambda fn, *a: fn(*a), lambda _: None, done, lambda _: None)
        try:
            worker.enable(True)
            worker.write(b'old host', b'guest', 3)
            self.assertTrue(event.wait(2))
            self.assertEqual(applied, [(3, False, b'newer host')])
            self.assertEqual(fake.writes, [])
        finally: worker.close()


if __name__ == '__main__': unittest.main()
