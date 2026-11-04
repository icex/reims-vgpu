#!/usr/bin/env python3
"""Bounded text-only SPICE clipboard using unfocused Wayland data-control tools.

No clipboard payload is printed or written to disk. One worker serializes bounded
wl-paste reads and conditional wl-copy writes; the GTK loop only handles SPICE
signals. Repeated snapshots and our own clipboard writes never become new grabs.
"""
import ctypes
import os
import queue
import select
import subprocess
import threading
import time

LIMIT = 65536
SELECTION = 0
UTF8 = 1


def text(data):
    if not isinstance(data, bytes) or len(data) > LIMIT or b'\0' in data:
        raise ValueError('clipboard text refused')
    data.decode('utf-8', 'strict')
    return data


def bounded_read(argv, timeout=1):
    """Limit bytes while reading, rather than after subprocess captured them."""
    process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL)
    result = bytearray()
    deadline = time.monotonic() + timeout
    try:
        os.set_blocking(process.stdout.fileno(), False)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('clipboard provider timeout')
            ready, _, _ = select.select([process.stdout], [], [], min(.05, remaining))
            if not ready:
                continue
            part = os.read(process.stdout.fileno(), LIMIT + 1 - len(result))
            if not part:
                break
            result.extend(part)
            if len(result) > LIMIT:
                raise ValueError('clipboard provider limit')
        code = process.wait(timeout=max(.01, deadline - time.monotonic()))
        return text(bytes(result)) if code == 0 else None
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=1)
        process.stdout.close()


class WaylandClipboard:
    """Only the regular text selection; never PRIMARY, images or file URLs."""
    def read(self):
        try:
            return bounded_read(['wl-paste', '--no-newline', '--type', 'text'])
        except (ValueError, UnicodeDecodeError):
            return None

    def write(self, data):
        text(data)
        # Contents travel through stdin, never shell text or process arguments.
        subprocess.run(['wl-copy', '--type', 'text/plain;charset=utf-8'], input=data,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=1, check=True)


class Worker:
    """One in-flight backend operation, callbacks delivered on the GTK thread."""
    def __init__(self, backend, dispatch, changed, applied, failed):
        self.backend = backend
        self.dispatch = dispatch
        self.changed = changed
        self.applied = applied
        self.failed = failed
        self.commands = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.enabled = False
        self.thread = threading.Thread(target=self._run, name='reims-clipboard', daemon=True)
        self.thread.start()

    def enable(self, enabled):
        self.enabled = enabled

    def write(self, expected, data, generation):
        try:
            self.commands.put_nowait((expected, text(data), generation))
        except queue.Full:
            self.dispatch(self.applied, generation, False, expected)

    def _run(self):
        while not self.stop_event.is_set():
            try:
                command = self.commands.get(timeout=.4)
            except queue.Empty:
                command = None
            if not self.enabled:
                if command is not None:
                    self.dispatch(self.applied, command[2], False, command[0])
                continue
            try:
                current = self.backend.read()
                if command is None:
                    self.dispatch(self.changed, current)
                    continue
                expected, data, generation = command
                # A newer host copy wins over an old guest response. Wayland has
                # no atomic compare-and-set; this is the smallest read/write gap.
                success = current == expected and self.enabled
                if success:
                    self.backend.write(data)
                    current = data
                self.dispatch(self.applied, generation, success, current)
            except (OSError, ValueError, TimeoutError, subprocess.SubprocessError) as error:
                self.dispatch(self.failed, type(error).__name__)
                if command is not None:
                    self.dispatch(self.applied, command[2], False, command[0])

    def close(self):
        self.enabled = False
        self.stop_event.set()
        self.thread.join(timeout=3)


class Clipboard:
    """SPICE by-demand state machine; all methods run on the GTK loop."""
    def __init__(self, channel, write, record=lambda message: None, clock=time.monotonic):
        self.channel = channel
        self.write = write
        self.record = record
        self.clock = clock
        self.enabled = False
        self.connected = False
        self.baseline = False
        self.host = None
        self.owned = False
        self.generation = 0
        self.inflight = None
        self.want = False
        self.disabled = False

    def set_enabled(self, value):
        self.enabled = bool(value)
        self._reset()
        if not self.enabled and self.connected:
            self.channel.clipboard_selection_release(SELECTION)

    def set_connected(self, value):
        self.connected = bool(value)
        if not self.connected:
            self.inflight = None
        self._reset()

    def _reset(self):
        self.generation += 1
        self.baseline = False
        self.owned = False
        self.want = False
        # A response without a request ID must be drained before another request.
        # Keep inflight across toggle changes. Reconnect creates a new instance.
        self.disabled = False

    def active(self):
        return self.enabled and self.connected and not self.disabled

    def host_changed(self, current):
        if not self.active():
            return False
        if current is not None:
            text(current)
        if not self.baseline:
            self.host = current
            self.baseline = True
            self._request()
            return False
        if current == self.host:
            return False
        self.host = current
        self.generation += 1
        self.want = False
        self.owned = current is not None
        if self.owned:
            self.channel.clipboard_selection_grab(SELECTION, [UTF8])
        else:
            self.channel.clipboard_selection_release(SELECTION)
        return False

    def requested(self, _channel, selection, kind):
        if selection != SELECTION:
            return False
        if self.active() and kind == UTF8 and self.owned and self.host is not None:
            self.channel.clipboard_selection_notify(SELECTION, UTF8, list(self.host))
            return True
        self.channel.clipboard_selection_notify(SELECTION, 0, [])
        return False

    def grabbed(self, _channel, selection, types, count):
        if selection != SELECTION:
            return False
        if not self.active():
            return True
        if not 0 <= count <= 64:
            return True
        # GObject signals expose gpointer as an address, unlike GI methods.
        if isinstance(types, int):
            offered = list((ctypes.c_uint32 * count).from_address(types)) if count and types else []
        else:
            offered = list(types or [])[:count]
        self.generation += 1
        self.owned = False
        self.want = UTF8 in offered
        self._request()
        return True

    def _request(self):
        if self.active() and self.baseline and self.want and self.inflight is None:
            self.inflight = (self.generation, self.host, self.clock())
            self.channel.clipboard_selection_request(SELECTION, UTF8)

    def released(self, _channel, selection):
        if selection == SELECTION:
            self.generation += 1
            self.want = False
        # A release isn't empty text: don't erase the host clipboard.

    def received(self, _channel, selection, kind, data, size):
        if selection != SELECTION or self.inflight is None:
            return
        generation, expected, _ = self.inflight
        self.inflight = None
        if self.active() and self.want and generation == self.generation:
            # Refusal consumes this offer; do not request an unsupported offer forever.
            self.want = False
            if kind != UTF8 or not 0 <= size <= LIMIT:
                return
            try:
                value = ctypes.string_at(data, size) if isinstance(data, int) else bytes(data or b'')[:size]
                text(value)
                if len(value) != size:
                    raise ValueError('clipboard signal size')
            except (ValueError, UnicodeDecodeError):
                self.want = False
                self.record('clipboard-refused=text-shape')
            else:
                self.want = False
                self.write(expected, value, generation)
        self._request()

    def applied(self, generation, success, current):
        if not self.active():
            return False
        if success:
            self.host = current
            self.baseline = True
            self.owned = False
            self.record('clipboard-guest-applied bytes=' + str(len(current)))
        else:
            self.host_changed(current)
        return False

    def tick(self):
        if not self.disabled and self.inflight is not None and self.clock() - self.inflight[2] > 5:
            self.disabled = True
            self.want = False
            self.owned = False
            self.channel.clipboard_selection_release(SELECTION)
            self.record('clipboard-disabled=response-timeout')
        return True
