# Local desktop saved machines, RAM pause and host sleep

## Disk-backed saved machines

The `disk_save: true` launcher policy saves RAM and devices to SSD/HDD on X,
then exits QEMU. The shortcut restores the same session in a new QEMU process.
`save-state=on` is explicit; the graphics migration blocker remains for other
builds/policies. The local launcher pins the measured invariant-TSC frequency.

`saved-machine.py` waits for migration completion, fsyncs the image and manifest,
then quits the bound QEMU process. Before commit, write errors preserve the
source in RAM and recover its GPU state. The window minimizes immediately on X.
The next successful save removes only the preceding restored image, retaining
one current checkpoint. Image/manifest permissions are private to the account.

Before restore, exact QEMU build/configuration, unchanged disk identities and
image size/hash must match. A damaged or interrupted restore never silently
falls back to a cold boot. The restore-in-progress marker is durable before
vCPUs resume. Repeated shortcut opens are serialized. Full shutdown remains a
separate action; restore a saved session before requesting a guest shutdown.

The patch suspends GPU tasks before copying RAM, includes mapper IDs and
host-only surface bytes, and saves USB audio and AppleSMC protocol state.
A failed save detaches the retained GPU root before accepting its saved
registers; otherwise teardown would clear restored FIFO/stamp state.

Observed on2026-11-05: a direct macOS process-exit/restore, three native X /
shortcut cycles, and two further cycles after a real injected manifest-write
failure. The latter recovered the same source VM and accepted a new keyboard
edit. Cycles retained guest boot, WindowServer/app PIDs, unsaved Cocoa text,
fresh edits and rendered native-window output. One earlier cycle with unexpected
extra keyboard input was excluded; the window now minimizes at save start and
the harness restores prior host focus. Original disk identities stayed unchanged
through disposable-clone tests. No physical USB/audio or host-reboot claim.

The installed bundle is retained under
`~/.local/lib/reims-vgpu/saved-machine-20261105`; build caches are separate.
`local-launcher/` records this machine's installed launcher/settings. The
host remains free to sleep: there is no blocking sleep inhibitor, only a short
logind preparation delay while a live guest is paused. A saved guest holds no
QEMU process or sleep inhibitor. Guest wall-clock correction after time spent
saved is not qualified and remains a known limitation without guest tools.

Focused tests: `test_saved_machine.py`, `test_saved_icon.py`, `test_sleep.py`,
`test_close_guest.py`, `test_session_close.py`; mapper/core regression tests;
exact-binary emulated-device and macOS roundtrips. Evidence is in
`evidence/codex-20261105-disk-save/` in the parent workspace.

## RAM-only compatibility policy


The desktop shortcut starts macOS or resumes its retained RAM session.
Native window X selects QEMU `close-action=pause`: QEMU pauses its CPUs,
keeps its RAM/devices alive, and notifies `close-guest.py`. The helper
minimizes the existing KDE window. Reopening the shortcut continues QEMU
and restores that same window. No ACPI power button, guest S3, SDL hide/show,
or shutdown is used for X. Closing repeatedly never forces power off.

This is RAM pause, not disk hibernation. Ending QEMU, powering off, or
rebooting the host loses unsaved RAM state. Use the shortcut's explicit
"Request macOS shutdown" action for ordinary shutdown and save dialogs.
`close-events.py` retains the key-only SSH shutdown fallback and identity
checks; only an explicit shutdown request selects it.

Install `close-guest.py`, `close-events.py`, and `sleep-control.py` beside
`run-session.py`. Install `reims-macos` executable in `~/.local/bin` and
`reims-macos.desktop` in the desktop/application directories. The best
launcher must expose both `RUN/qmp.sock` and `RUN/close-events.sock`, and
set `close_action` to `pause`. Apply `../qemu-ram-pause.patch` to the pinned
QEMU source; `../build.sh` applies it and runs its policy check.
`local-launcher/` preserves this machine's exact installed launcher/settings;
its build paths are local and must be adjusted when using another bundle.

The VM launcher must run `python3 -B run-session.py RUN` directly, without
`systemd-inhibit --mode=block`. The helper's logind **delay** inhibitor lets
normal host sleep proceed after it pauses the guest; it does not block sleep.
On host wake a previously running guest resumes automatically. A guest
already paused with X stays paused until the shortcut is opened. Monitor
idle/sleep is allowed. Requires KDE KWin scripting, qdbus6, journalctl,
Python dbus/GLib and notify-send. Other desktops need their own window control.

Validation on 2026-11-05: ten native-close/shortcut-resume cycles retained
an authored unsaved Cocoa text window, guest/WindowServer/application PIDs,
keyboard editing, and the same native host window. Physical host suspend
measured 23.888 seconds; the same guest/app processes and recorded contents
survived, and QEMU resumed automatically. The post-wake fixture window was
not on screen, so further interactive fixture checks were not qualified.
**Known limitation:** the guest clock remained about 30 seconds behind
following that host cycle; NTP query confirmed the offset. Automatic clock
resynchronization is not qualified. Guest administrator credentials were
not accessed. Detailed evidence: `evidence/codex-20261105-sleep/` in the
parent workspace. This does not qualify disk hibernation, host reboot
session retention, or sustained rendering performance.

Focused tests (no VM/SSH/GPU launch): `python3 -B test_sleep.py`,
`python3 -B test_close_guest.py`, `python3 -B test_session_close.py`.

The optional `desktop-controls.py` connects the private SPICE text clipboard.
On Wayland it uses installed `wl-paste`/`wl-copy` data-control access, so the
native Reims window may retain focus. It does not instantiate GtkSession on
Wayland: auto-clipboard=False still installs GTK request handlers whose private
clip_grabbed state does not track data-control grabs. MainChannel alone owns the
Wayland clipboard; X11 retains GtkSession's clipboard.
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

## Home, clipboard and native drops

The desktop supervisor accepts three independent booleans in its adjacent
`desktop-settings.json`: `home_share`, `clipboard`, and `file_drop`. Enable all
three for desktop integration. Older settings keep home sharing and drops off;
text clipboard sharing retains its existing default. The controls window also
has a live clipboard checkbox. On Wayland, only the data-control bridge owns
clipboard signals; X11 uses the GTK clipboard client.

`home-share.py` starts the locally available Samba image on host loopback only
and configures the guest's removable login agent over the private SSH forward.
Finder's ~/Linux Home link opens the writable host home. Dotfiles are
visible with Cmd+Shift+period. The guest can read host credentials including
`~/.ssh`; storage vetoes and read-only masks protect the VM's disks. See
HOME-SHARE.md for exact protection semantics and removal.

With file_drop enabled the supervisor creates a mode0600 datagram socket in its
mode0700 run directory before QEMU starts, then sets REIMS_VGPU_DROP_SOCKET.
Without this opt-in the native window has its existing behaviour. SDL drop
paths are data, never shell code; the helper copies files/folders over SSH to
macOS Desktop, preserves symlinks, selects an unused name, and reveals the copy
in Finder. Host notifications show copying, completion or failure. Copying to
the Finder location under the pointer and dragging files out of macOS are not
provided. Auxiliary failures leave QEMU running; their logs are in the run dir.

The guest clipboard bootstrap packages only authored sources, builds its small
AppKit helper inside macOS, and retires the legacy clipboard job while keeping
its files and prior state for rollback. See guest-clipboard/README.md.

Dropped files land in macOS Desktop and can sync to iCloud. The home-share
shortcut stays at ~/Linux Home, outside Desktop/Documents/iCloud Drive.


Saved-machine operations show a non-cancelling Zenity progress window. Saving
reports QEMU's written/remaining memory counts, then flushing, verification and
closing phases. Restore uses an animated bar while validation and incoming
state loading run; it does not invent a percentage for phases QEMU cannot
measure. On KDE the feedback stays above the incoming VM window. Closing the
feedback window does not cancel the operation. Missing/closed UI never changes
the durable-save transaction or guest recovery behavior. Requires `zenity`.
