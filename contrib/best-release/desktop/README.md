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
