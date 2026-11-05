#!/usr/bin/python3
"""Non-cancelling desktop feedback; closing the dialog never cancels VM I/O."""
import subprocess
import threading
import runpy
from pathlib import Path


class Progress:
    def __init__(self, title):
        self.process = None
        try:
            self.process = subprocess.Popen(
                ['zenity', '--progress', '--title=' + title, '--text=Preparing…',
                 '--pulsate', '--no-cancel', '--auto-close', '--width=440'],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, text=True)
            threading.Thread(target=self._raise, daemon=True).start()
        except OSError:
            pass  # Desktop feedback must never break a save or restore.

    def _raise(self):
        # Keep feedback visible when the incoming QEMU window first appears.
        try:
            window = runpy.run_path(str(Path(__file__).with_name('sleep-control.py')))['window']
            for _ in range(8):
                process = self.process
                if process is None or process.poll() is not None:
                    return
                try:
                    window(process.pid, 'progress')
                    return
                except Exception:
                    threading.Event().wait(.25)
        except Exception:
            pass

    def update(self, message):
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.stdin.write('# ' + message.replace('\n', ' ') + '\n')
                self.process.stdin.flush()
            except (OSError, ValueError):
                pass

    def close(self):
        if self.process is not None:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.process = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
