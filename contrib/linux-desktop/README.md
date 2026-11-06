# Linux desktop integration work in progress

This is development work, not a qualified desktop release. Nothing in this
directory runs at boot unless explicitly installed or launched.

## Source identity and QEMU patch

The QEMU submodule stays pinned to its published upstream commit
`bd95509aa307f48b13fe8497d8a1b64267cb4c3e`.

`patches/qemu-desktop.patch` carries the QEMU changes as source, avoiding a
submodule pointer to an unpublished commit. It adds variable-length USB audio
packets, GTK aspect-preserving scaling and optional aspect snapping, auto-fit,
host cursor selection, and a USB menu request. SDL/native Vulkan window behavior
already present upstream is not reimplemented here. No VFIO or iGPU DMA changes
are included. Apply only to a private QEMU checkout at the pinned upstream commit:

```sh
git -C /path/to/private/qemu apply --check /path/to/qemu-desktop.patch
git -C /path/to/private/qemu apply /path/to/qemu-desktop.patch
```

The patch in this directory has been checked for applicability at the pinned
commit only (`git apply --check`); it has not been built or run from here. GTK
behavior and USB/audio still require a build and their manual regression walk.
The native Vulkan window does not exercise the GTK rendering changes.

## Display modes and guest helpers

Configure the development QEMU GPU with explicit refresh choices; a mode list
containing only 120 Hz gives Displays no alternative rate to show. For example,
these are two separate JSON `-device` arguments (replace the ROM path):

```json
{"driver":"apple-gfx-pci","id":"reimsvgpu","bus":"pcie.0","addr":"0x6","romfile":"/path/to/ReimsVgpu.rom","resolution":"3840x2160@220","display-modes":["3840x2160@240","3840x2160@120","3840x2160@60"]}
{"driver":"apple-gfx-iosurface-pci","bus":"pcie.0","addr":"0x7"}
```

Use JSON for both devices to preserve realization order. The IOSurface mapper is
for the development guest-driver stack. Use the matching upstream reims-util /
OpenCore generation procedure; this directory supplies no guest binary or EFI image.

Compile and invoke the guest display utility manually in a logged-in Aqua session:

```sh
clang -O2 -fobjc-arc -Wno-deprecated-declarations guest/display-modes.m \
  -o display-modes -framework Foundation -framework AppKit \
  -framework CoreGraphics -framework CoreVideo
./display-modes
./display-modes 1920 1080 3840 2160 240
```

With no arguments it reports modes. Selection persists the chosen mode. Reported
refresh and CVDisplayLink timing do not prove physical host scanout or sustained
frame delivery. The previous 120 Hz experiment showed a steady-window mean of
119 submitted frames/s, with dips; 240 Hz delivery is not qualified.

`guest/compute-validation.m` is an optional manual MPS FP32 matrix check. It is
never installed as a startup job:

```sh
clang -O2 -fobjc-arc guest/compute-validation.m -o compute-validation \
  -framework Foundation -framework Metal -framework MetalPerformanceShaders \
  -framework QuartzCore
./compute-validation
```

One macOS 15.7.9 run passed its 256 CPU reference checks with no nonfinite output.
This single workload is not full Metal qualification or a macOS 26 result.

## Optional text clipboard and USB controls

`guest/install-clipboard.py` explicitly installs only the per-user clipboard agent.
Review it and invoke it as the logged-in guest user if sharing is wanted. It copies
the agent, the modules it loads and the pasteboard helper's source into
`~/Library/Application Support/Reims`, compiles the helper, and installs a
persistent per-user LaunchAgent, `org.reims.clipboard` (`RunAtLoad` and
`KeepAlive`), which starts at every login until removed:

```sh
launchctl bootout gui/$(id -u)/org.reims.clipboard
rm ~/Library/LaunchAgents/org.reims.clipboard.plist
```

It needs Python 3, Xcode command-line tools, and a virtio-serial channel named
`com.redhat.spice.0` connected to QEMU's SPICE vdagent. It does not configure login
passwords, sleep policy, display overrides, or any benchmark startup job. The
agent's two logs, `clipboard.log` and `clipboard-error.log` in that directory,
grow without bound; nothing rotates them.

The guest agent bounds text to 64 KiB and starts from a clipboard baseline. It also
answers SPICE monitor-resize requests, which it forwards to a per-user display
holder on a control socket in `Reims/control`. This directory ships no such holder,
so without one every resize request is refused.

Native Vulkan display and clipboard are separate: `host/desktop-controls.py
RUN_DIRECTORY` connects only to the private SPICE socket. It disables the audio,
USB-redirection and smartcard channels, turns automatic USB redirection off, and
opens no display widget: the native Vulkan window shows the guest. It needs GTK 3,
PyGObject and spice-gtk introspection. Use a caller-owned run directory with mode
0700 and `spice.sock` inside it. A change to `usb-picker.request` in that directory
after it starts opens the USB picker.

For USB, set `RGPU_RUN_DIR` explicitly to that VM's private run directory. QEMU
must provide two initially empty usb-redir sockets named `usbredir-0.sock` and
`usbredir-1.sock`. The helper uses user systemd services and `usbredirect`.
No physical USB device is selected automatically. Devices are named by VID:PID,
so `host/reims-usb.sh attach` refuses when more than one attached device has the
same VID:PID, and the picker does not offer such devices. The helpers' service names
support a single Reims control instance per user; multiple concurrent Reims
instances need separate service namespaces. They do not access other VMs.

## Hardware video prototype and remaining work

`host/host-video-offload.py` manually streams a guest file over SSH through host
VAAPI and returns a new MP4. Set up a guest SSH alias and verify its host key first:

```sh
python3 host/host-video-offload.py input.mp4 output.mp4 \
  --ssh-host reims-guest --render-node /dev/dri/renderD128 --codec hevc
```

Inputs and outputs are filenames under the guest user's
`~/Library/Application Support/Reims/video-tests/`. The input reaches ffmpeg
through a pipe, so an MP4 input must be faststart (its `moov` atom before the
media data; `ffmpeg -i in.mp4 -c copy -movflags +faststart out.mp4` rewrites one).
SSH identity and account configuration stay outside this repository. The
prototype starts no network service and does not register a macOS VideoToolbox
codec. An earlier version returned a valid 120-frame 4K HEVC file, with a
complete software decode; the configurable version here has only static
validation.

Native hardware-required VideoToolbox sessions were unavailable in macOS 15.7.9
(encode -12908, decode -12906). Host VAAPI encode/decode working does not enable
ordinary macOS apps. A real port still needs a guest codec provider, a validated
VirtIO message contract, QEMU/backend support, shared-surface lifetime and bounds
checks, asynchronous completion/flush/teardown, and unchanged native VT probes.

## Crash state and qualification limits

Weather's Metal archive abort stopped recurring during a 60-second check
with the upstream-documented `METAL_ERROR_MODE=5` preference. This is an explicit
guest configuration step, not a rendering or hardware-codec implementation:

```sh
sudo defaults write /Library/Preferences/.GlobalPreferences METAL_ERROR_MODE -int 5
sudo chmod 644 /Library/Preferences/.GlobalPreferences.plist
```

A separate Sequoia WindowServer tiled-texture failure was investigated with a
private guest-specific compatibility patch, which is not included here. The
macOS 26 guest-driver path is untested with these helpers.

Both boots with the 240 Hz mode list and that tile-check patch ended about 50 s
after start: the host library aborted when metal2vulkan refused a system RenderBox
fragment shader (`Unsupported { feature: Lowering(StageIo) }`), whose
`[[front_facing]]` parameter macOS 15's compiler emits as an `i8`. The stable and
development backends fail identically; a metal2vulkan fix is in progress. Earlier
sessions without those changes (85 min at 1080p60, 39 min at 4K/120) did not hit
it. Do not infer desktop stability.

No GPU fault suite or performance qualification covers these helpers. Launcher
settings, VM disks, firmware images, guest binaries/shaders, account setup,
passwords, SSH keys, machine-specific configuration, screenshots and raw runtime
logs are not part of this directory.


The USB picker labels storage and Bluetooth adapters using their sysfs interface
classes and product strings. A permission failure for another selected device
is independent of a successfully redirected drive. Forwarding a drive does not
mean its filesystem has finished mounting: APFS may validate checkpoints before
Finder can use a backup volume. Avoid disconnecting a drive during that check.
