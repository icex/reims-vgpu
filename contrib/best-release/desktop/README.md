# Local desktop clean-close integration

These scripts are the local desktop fork companion to the QEMU close-action
property. Install close-guest.py next to run-session.py and the desktop controls;
the reims-macos script is the start/stop/status/USB desktop entry point.

A best-channel launcher adds a dedicated second monitor:
`-qmp unix:RUN/close-events.sock,server=on,wait=off`.
The regular `RUN/qmp.sock` remains the icon/control connection. Select the
apple-gfx `close-action=powerdown` fallback when guest S3 is refused. The
supervisor derives a unique localhost guest SSH forward and disk from its
known QEMU argv and binds the helper to its actual child PID/executable/disk.
The guest SSH key defaults to `~/macos-vm/vmkey`; key bytes are never packaged.

POWERDOWN also requests ordinary guest shutdown over local SSH because a
guest policy may ignore its ACPI power button. The companion never terminates
QEMU, discards documents or requests host sleep. Save dialogs may delay
shutdown; finish them inside the guest. Duplicate pending closes are coalesced.

Pure socket/supervisor tests: `python3 -B test_close_guest.py` and
`python3 -B test_session_close.py`. They launch no QEMU, SSH or GPU workload.
The hardware gate must verify native close callback, guest SHUTDOWN event,
QEMU exit, next-boot filesystem state and host-suspend safety separately.


The optional `desktop-controls.py` connects the private SPICE text clipboard.
On Wayland it uses installed `wl-paste`/`wl-copy` data-control access, so the
native Reims window may retain focus. It disables GtkSession auto-clipboard on
Wayland to keep a single clipboard owner. X11 retains GtkSession's clipboard.
Copying an image, file selection, invalid UTF-8, NUL or more than 65536 bytes
releases the text offer without clearing the other clipboard. An empty text
string is distinct from releasing a selection. Only new text changes after the
initial baseline are shared. Clipboard contents are never logged or written to
disk by these helpers.

The controls checkbox changes sharing immediately. Adjacent
`desktop-settings.json` may set `"clipboard": false` for the next launch;
`REIMS_CLIPBOARD=0` also disables initial sharing. `wayland-clipboard.py` must
be installed alongside `desktop-controls.py`. Reads run on one worker, are
bounded during transport and time out. A new host copy supersedes a pending
guest response. Repeated snapshots and applied guest text are not reoffered.
SPICE has no request IDs; an ambiguous five-second response timeout disables
sharing until the channel reconnects rather than accepting a late reply as a
new clipboard. No guest pasteboard is accessed over SSH.

No-hardware checks:
`python3 -m unittest discover -s contrib/best-release/desktop -p test_wayland_clipboard.py`.
The unit checks do not establish compositor focus access, guest pasteboard
behavior or keyboard capture; exercise both real pasteboards on a VM for those.
