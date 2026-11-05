#!/usr/bin/env python3
"""Launch updated Reims on an independent disk without touching Raphael."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import runpy
import subprocess

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'baseline'
BUILD = Path('/home/bogdan/.cache/reims-desktop-qemu-20261017')
DIST = Path('/home/bogdan/.cache/reims-desktop-target-20261017/dist-vulkan')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument('--resume', type=Path, help='resume only an existing independent Reims run')
    choice.add_argument('--fresh', action='store_true', help='create another test copy from the old baseline')
    parser.add_argument('--gtk', action='store_true', help='use the patched GTK console instead of the native Vulkan window')
    parser.add_argument('--resolution', help='guest pixels and optional density, e.g. 3840x2160@220 or 2560x1600@200')
    parser.add_argument('--refresh-rate', type=int, choices=(60, 120, 240), help='offered guest refresh rate in Hz')
    parser.add_argument('--channel', choices=('stable', 'development', 'candidate', 'test', 'daily', 'best'), help='select an already built local Reims version')
    parser.add_argument('--dry-run', action='store_true', help='print the selected argv and environment without starting or changing a VM')
    options = parser.parse_args()
    if options.dry_run and options.fresh:
        parser.error('--dry-run requires an existing run')
    settings_path = ROOT / 'desktop-settings.json'
    settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
    resolution = options.resolution or settings.get('resolution', '3840x2160@220')
    refresh = options.refresh_rate or settings.get('refresh_hz', 120)
    channel = options.channel or settings.get('channel', 'stable')
    halt_poll_ns = settings.get('halt_poll_ns') if channel == 'best' else None
    if halt_poll_ns is not None and (type(halt_poll_ns) is not int or not 0 <= halt_poll_ns < 2**64):
        parser.error('halt_poll_ns must be an unsigned integer, or absent for the KVM default')
    close_action = settings.get('close_action', 'suspend') if channel == 'best' else None
    if channel == 'best' and close_action not in ('pause', 'suspend', 'powerdown'):
        parser.error('close_action must be pause, suspend or powerdown')
    build, dist, manifest = {
        'best': (Path('/home/bogdan/.local/lib/reims-vgpu/saved-machine-20261105/bin'),
            Path('/home/bogdan/.local/lib/reims-vgpu/saved-machine-20261105'),
            '/home/bogdan/.local/lib/reims-vgpu/saved-machine-20261105/manifest.json'),
        'daily': (Path('/home/bogdan/.cache/reims-proven-fixes-20261024/bundle/bin'),
            Path('/home/bogdan/.cache/reims-proven-fixes-20261024/bundle'),
            '/home/bogdan/.cache/reims-proven-fixes-20261024/bundle/manifest.json'),
        'stable': (BUILD, DIST, 'manifest.json'),
        'development': (Path('/home/bogdan/.cache/reims-desktop-dev-qemu-20261017'),
            Path('/home/bogdan/.cache/reims-desktop-dev-target-20261017/dist-vulkan'), 'development-manifest.json'),
        'candidate': (Path('/var/tmp/reims-candidate-qemu'),
            Path('/var/tmp/reims-candidate-target/dist-vulkan'), 'candidate-manifest.json'),
        # A build under test, configured like candidate, beside it: benchmarks boot it, users do not.
        'test': (Path('/var/tmp/reims-test-qemu'),
            Path('/var/tmp/reims-test-target/dist-vulkan'), 'candidate-manifest.json'),
    }[channel]
    manifest = ROOT / 'patches' / manifest
    match = re.fullmatch(r'([1-9][0-9]*)x([1-9][0-9]*)(?:@([1-9][0-9]*))?', resolution)
    if resolution != 'host' and (not match or any(int(v) > 65535 for v in match.groups() if v)):
        parser.error('resolution must be widthxheight, widthxheight@ppi or host')
    if refresh not in (60, 120, 240):
        parser.error('refresh rate must be 60, 120 or 240 Hz')
    if not options.dry_run:
        os.umask(0o077)
        lock = (ROOT / 'launch.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if subprocess.run(['systemctl', '--user', 'is-active', '--quiet', 'reims-9070xt.service']).returncode == 0:
            raise SystemExit('Reims is already running')
        if Path('/sys/bus/pci/devices/0000:03:00.0/driver').resolve().name != 'amdgpu':
            raise SystemExit('RX 9070 XT must remain on amdgpu')
    # Guest size: REIMS_VCPUS and REIMS_RAM_GB, else the saved settings, else 4 vCPUs and 6 GiB.
    vcpus = int(os.environ.get('REIMS_VCPUS', settings.get('vcpus', 4)))
    ram_gb = int(os.environ.get('REIMS_RAM_GB', settings.get('ram_gb', 6)))
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))) * 1024
    if available < (ram_gb + 1) * 1024**3:
        raise SystemExit(f'Need {ram_gb + 1} GiB available before starting a separate {ram_gb} GiB guest')
    qemu = build / 'qemu-system-x86_64'
    rom = dist / 'share/qemu/ReimsVgpu.rom'
    if not rom.is_file():
        found = list(dist.rglob('ReimsVgpu.rom'))
        if len(found) == 1:
            rom = found[0]
    for path in (qemu, rom, manifest):
        if not path.is_file():
            raise SystemExit('Build missing: ' + str(path))
    if not options.resume and not options.fresh:
        saved_run = Path(settings['run']) if settings.get('run') else ROOT / 'current'
        if (saved_run / 'latest-build.json').is_file() and (saved_run / 'macos.qcow2').is_file():
            options.resume = saved_run
    if options.resume:
        run = options.resume.resolve(strict=True)
        if run.parent != (ROOT / 'runs').resolve() or not (run / 'latest-build.json').exists():
            raise SystemExit('Resume requires an updated independent Reims run')
    else:
        run = ROOT / 'runs' / datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        run.mkdir(mode=0o700)
        for name in ('macos.qcow2', 'OpenCore.qcow2', 'OVMF_VARS.fd'):
            subprocess.run(['cp', '--reflink=auto', str(BASE / name), str(run / name)], check=True)
            (run / name).chmod(0o600)
    gpu_device = dict(driver='apple-gfx-pci', id='reimsvgpu', bus='pcie.0',
        addr='0x6', romfile=str(rom), resolution=resolution)
    if close_action is not None:
        gpu_device['close-action'] = close_action
        if settings.get('disk_save'):
            gpu_device['save-state'] = True
    if match:
        width, height = int(match[1]), int(match[2])
        sizes = dict.fromkeys([(width, height), (3840,2160), (2560,1600),
            (2560,1440), (1920,1200), (1920,1080), (1600,900), (1440,900),
            (1280,800), (1280,720)])
        rates = dict.fromkeys((refresh, 60, 120, 240))
        # The host monitor's 7680x2160 and the 5K sizes are offered whatever the first mode, for
        # HiDPI 3840x1080, 2560x1440 and 2560x720 desktops. The paravirtual display offers no
        # scaled modes, so the 32:9 'more space' and 'larger text' desktops are pixel modes of
        # their own (HiDPI 5120x1440 and 3200x900), which the window scales to its size.
        large = [(10240,2880), (7680,2160), (6400,1800), (5120,2880), (5120,1440)]
        gpu_device['display-modes'] = [f'{w}x{h}@{hz}' for w,h in sizes
            if w <= width and h <= height for hz in rates]
        gpu_device['display-modes'] += [f'{w}x{h}@{hz}' for w,h in large
            if (w, h) not in sizes or w > width or h > height for hz in rates]
    cpu_model = 'Haswell-noTSX,kvm=on,vendor=GenuineIntel,+invtsc,vmware-cpuid-freq=on'
    if settings.get('disk_save'):
        frequency = settings.get('tsc_frequency_hz')
        if type(frequency) is not int or frequency <= 0 or frequency % 1000:
            parser.error('disk_save requires a measured integer tsc_frequency_hz in whole kHz')
        cpu_model += f',tsc-frequency={frequency}'
    args = [str(qemu), '-L', '/usr/share/qemu', '-name', 'Reims macOS - RX 9070 XT', '-enable-kvm', '-m', f'{ram_gb}G',
        '-object', f'memory-backend-memfd,id=reims-ram,size={ram_gb}G,share=on',
        '-machine', 'q35,memory-backend=reims-ram',
        '-cpu', cpu_model,
        '-smp', f'{vcpus},sockets=1,cores={vcpus},threads=1',
        '-device', 'qemu-xhci,id=xhci', '-global', 'qemu-xhci.p2=8', '-global', 'qemu-xhci.p3=8',
        '-device', 'usb-kbd,bus=xhci.0', '-device', 'usb-tablet,bus=xhci.0',
        '-device', 'isa-applesmc,osk=ourhardworkbythesewordsguardedpleasedontsteal(c)AppleComputerInc',
        '-drive', f'if=pflash,format=raw,readonly=on,file={BASE}/OVMF_CODE.fd',
        '-drive', f'if=pflash,format=raw,file={run}/OVMF_VARS.fd', '-smbios', 'type=2',
        '-device', 'ich9-ahci,id=sata',
        '-drive', f'id=OpenCoreBoot,if=none,format=qcow2,file={run}/OpenCore.qcow2',
        '-device', 'ide-hd,bus=sata.2,drive=OpenCoreBoot',
        '-drive', f'id=MacHDD,if=none,format=qcow2,file={run}/macos.qcow2',
        '-device', 'ide-hd,bus=sata.4,drive=MacHDD',
        '-netdev', 'user,id=net0,ipv6=off,hostfwd=tcp:127.0.0.1:10023-:22,hostfwd=tcp:127.0.0.1:5901-:5900',
        '-device', 'vmxnet3,netdev=net0,id=net0,mac=52:54:00:90:70:01',
        '-vga', 'none', '-device', json.dumps(gpu_device),
        '-display', 'gtk,gl=on,zoom-to-fit=on' if options.gtk else 'none',
        '-serial', f'file:{run}/serial.log', '-qmp', f'unix:{run}/qmp.sock,server=on,wait=off',
        '-spice', f'unix=on,addr={run}/spice.sock,disable-ticketing=on,disable-agent-file-xfer=on,agent-mouse=off',
        '-device', 'virtio-serial-pci,id=reims_agent_serial,bus=pcie.0,addr=0x10,max_ports=2,event_idx=off',
        '-chardev', 'spicevmc,id=reims_vdagent,name=vdagent',
        '-device', 'virtserialport,id=reims_agent_port,bus=reims_agent_serial.0,nr=1,chardev=reims_vdagent,name=com.redhat.spice.0',
        '-audiodev', 'pa,id=audio0,out.latency=80000,out.frequency=48000',
        '-device', 'usb-audio,audiodev=audio0,bus=xhci.0,buffer=49152']
    if channel == 'best':
        args += ['-qmp', f'unix:{run}/close-events.sock,server=on,wait=off']
    for slot in range(2):
        args += ['-chardev', f'socket,id=reims_usbredir{slot},path={run}/usbredir-{slot}.sock,server=on,wait=off',
                 '-device', f'usb-redir,id=reims_usbredir_dev{slot},chardev=reims_usbredir{slot},bus=xhci.0']
    if channel in ('candidate', 'test', 'daily', 'best'):
        # The native window shows every frame; SPICE (clipboard only) needs no live console copy.
        gpu_device['window-readback-interval'] = int(os.environ.get('REIMS_READBACK_MS', '600000'))
        args[args.index('-device', args.index('-vga')) + 1] = json.dumps(gpu_device)
        # Timestamped log and trace lines, for timing the present path when traces are on.
        args += ['-msg', 'timestamp=on']
    if channel in ('development', 'candidate', 'test', 'daily', 'best'):
        index = args.index('-device', args.index('-vga'))
        args[index + 2:index + 2] = ['-device', json.dumps(dict(driver='apple-gfx-iosurface-pci',bus='pcie.0',addr='0x7'))]
    if close_action == 'suspend':
        # Advertise an ACPI guest sleep button only for the suspend policy.
        args += ['-global', 'ICH9-LPC.x-sleep-button=on']
    if halt_poll_ns is not None:
        index = args.index('-enable-kvm')
        args[index:index + 1] = ['-accel', f'kvm,halt-poll-ns={halt_poll_ns}']
    if settings.get('disk_save'):
        saved = runpy.run_path(str(ROOT / 'saved-machine.py'))
        incoming = saved['validate_restore'](run, args)
        if incoming is not None:
            args += ['-S', '-incoming', 'file:' + str(incoming)]
    if not options.dry_run:
        (run / 'command.json').write_text(json.dumps(args, indent=2) + '\n')
        (run / 'latest-build.json').write_text(manifest.read_text())
        (run / 'build-binary.json').write_text(json.dumps(dict(channel=channel, executable=str(qemu)), indent=2) + '\n')
        # A test boot keeps the saved channel, so the next ordinary start is the user's build again.
        saved = settings.get('channel', 'stable') if channel == 'test' else channel
        settings = dict(settings, resolution=resolution, refresh_hz=refresh, channel=saved, vcpus=vcpus, ram_gb=ram_gb,
            run=settings.get('run', str(run)) if channel == 'test' else str(run))
        settings_path.write_text(json.dumps(settings, indent=2) + '\n')
        settings_path.chmod(0o600)
        (run / 'display-settings.json').write_text(json.dumps(settings, indent=2) + '\n')
        current = ROOT / 'current'
        current.unlink(missing_ok=True)
        current.symlink_to(run)
    unit = ['systemd-run', '--user', '--unit=reims-9070xt', '--collect',
        '--property=TimeoutStopSec=20', '--property=RuntimeMaxSec=infinity', f'--property=WorkingDirectory={run}',
        # The VM before other user work: a CPU weight 100x the default within the user's own
        # session, so compiles and other apps do not take its CPU (ananicy-cpp renices QEMU to 9).
        '--property=CPUWeight=10000', '--property=IOWeight=1000',
        f'--property=StandardOutput=append:{run}/qemu.log', f'--property=StandardError=append:{run}/qemu.log',
        '--setenv=REIMS_VGPU_WINDOW=1', '--setenv=REIMS_VGPU_STATS=1', f'--setenv=REIMS_VGPU_WINDOW_REFRESH={refresh}',
        '--setenv=MESA_VK_DEVICE_SELECT=1002:7550!', '--setenv=RGPU_SNAP_ASPECT=1']
    unit += [f'--setenv=RGPU_USB_REQUEST={run}/usb-picker.request']
    if channel in ('candidate', 'test', 'daily', 'best'):
        # Answer a Mac's discrete GPU (no Apple family, no tile memory): macOS 26.4+ composites
        # through tile dispatches and framebuffer fetch otherwise, which Vulkan runs slowly.
        unit += [f"--setenv=REIMS_VGPU_APPLE_FAMILIES={os.environ.get('REIMS_VGPU_APPLE_FAMILIES', '1')}"]
        # Explicit diagnostics are supported only for experimental channels.
        if channel in ('candidate', 'test'):
            for key in ('REIMS_VGPU_LOG', 'REIMS_VGPU_TRACE', 'REIMS_VGPU_TRACE_DEGRADE', 'REIMS_VGPU_GPU_TIMESTAMPS', 'REIMS_PERF'):
                if key in os.environ:
                    unit += [f'--setenv={key}={os.environ[key]}']
    if channel in ('daily', 'best'):
        unit += [f'--setenv=LD_LIBRARY_PATH={dist}/lib', '--setenv=REIMS_DAILY_SESSION=1', '--setenv=REIMS_LAN=raphael']
    if channel == 'best':
        unit += ['--setenv=REIMS_VGPU_VRAM_SURFACES=1']
    for key in ('DISPLAY', 'WAYLAND_DISPLAY', 'XDG_RUNTIME_DIR', 'XAUTHORITY', 'DBUS_SESSION_BUS_ADDRESS', 'PULSE_SERVER'):
        if key in os.environ:
            unit += [f'--setenv={key}={os.environ[key]}']
    if options.dry_run:
        environment = dict(value.split('=', 2)[1:] for value in unit if value.startswith('--setenv='))
        print(json.dumps({'channel': channel, 'argv': args, 'environment': environment}, indent=2))
        return
    subprocess.run(unit + ['python3', '-B', str(ROOT / 'run-session.py'), str(run)], check=True)
    print(run)


if __name__ == '__main__':
    main()
