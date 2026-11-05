#!/usr/bin/python3
"""Transactional stopped-machine images; never exit QEMU before a durable save."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import time
import uuid

FORMAT = 1
SAVE_TIMEOUT = 300


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, 'w') as output:
            json.dump(value, output, indent=2)
            output.write('\n')
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def identity(path):
    path = Path(path).resolve(strict=True)
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
        raise RuntimeError('Saved-machine disk is not an owned regular file: ' + str(path))
    return dict(path=str(path), device=info.st_dev, inode=info.st_ino,
                size=info.st_size, mtime_ns=info.st_mtime_ns)


def base_command(args):
    result = []
    index = 0
    while index < len(args):
        if args[index] == '-incoming':
            index += 2
            continue
        if args[index] == '-S':
            index += 1
            continue
        result.append(args[index])
        index += 1
    return result


def drive_paths(args):
    paths = []
    for index, option in enumerate(args[:-1]):
        if option == '-drive':
            for value in args[index + 1].split(','):
                if value.startswith('file='):
                    paths.append(Path(value[5:]))
    if not paths:
        raise RuntimeError('No persistent disks found in saved-machine command.')
    return paths


def process_identity(pid):
    try:
        fields = Path('/proc', str(pid), 'stat').read_text().rsplit(') ', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return dict(boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                    start_ticks=fields[19])
    except FileNotFoundError:
        return None


def read_saved(run):
    path = run / 'saved-machine.json'
    if not path.exists():
        return None
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise RuntimeError('Saved-machine manifest has unsafe ownership or permissions.')
    result = json.loads(path.read_text())
    if not isinstance(result, dict) or result.get('format') != FORMAT:
        raise RuntimeError('Unsupported saved-machine manifest.')
    if result.get('phase') not in ('saved', 'restoring', 'restored'):
        raise RuntimeError('Invalid saved-machine phase; saved image preserved.')
    return result


def image_path(run, manifest):
    name = manifest['image']
    if Path(name).name != name or not name.startswith('saved-machine-') or not name.endswith('.bin'):
        raise RuntimeError('Invalid saved-machine image name.')
    result = run / name
    info = result.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise RuntimeError('Saved-machine image has unsafe ownership or permissions.')
    return result


def validate_restore(run, args):
    """Refuse changed disks/builds/configurations, never silently boot past a saved session."""
    manifest = read_saved(run)
    if manifest and manifest['phase'] == 'restoring':
        raise RuntimeError('The previous restore did not finish. Its image is preserved; do not cold-boot over it.')
    if not manifest or manifest['phase'] != 'saved':
        return None
    source = manifest['source_pid']
    if manifest.get('source_identity') is not None and process_identity(source) == manifest['source_identity']:
        raise RuntimeError('The source QEMU has not exited; wait for saving to finish.')
    if base_command(args) != manifest['command']:
        raise RuntimeError('The saved machine requires its original QEMU configuration.')
    if digest(args[0]) != manifest['qemu_sha256']:
        raise RuntimeError('The saved machine requires its exact QEMU build.')
    for disk in manifest['disks']:
        if identity(disk['path']) != disk:
            raise RuntimeError('A saved-machine disk changed: ' + disk['path'])
    image = image_path(run, manifest)
    if image.stat().st_size != manifest['image_size'] or digest(image) != manifest['image_sha256']:
        raise RuntimeError('The saved-machine image is incomplete or damaged.')
    return image


def save(run, qmp, source_pid, notify=lambda message: None):
    """QMP must be bound to the validated owned child. Failure keeps its RAM process alive."""
    run = run.resolve(strict=True)
    with (run / 'saved-machine.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior = read_saved(run)
        if prior and prior['phase'] == 'saved':
            raise RuntimeError('A saved session already exists; restore it before saving again.')
        args = base_command(json.loads((run / 'command.json').read_text()))
        required = 18 * 1024 ** 3
        if shutil.disk_usage(run).free < required:
            raise RuntimeError('At least 18 GiB of free space is required for a 16 GiB machine save.')
        name = 'saved-machine-' + uuid.uuid4().hex + '.bin'
        part = run / (name + '.partial')
        committed = False
        image = None
        migration_started = False
        fd = os.open(part, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        os.close(fd)
        try:
            if qmp.status() == 'running':
                qmp.command('stop')
            if qmp.status() != 'paused':
                raise RuntimeError('Disk save requires a paused machine outside guest S3.')
            notify('Saving macOS RAM and device state to disk…')
            qmp.command('migrate-set-parameters', {'max-bandwidth': 1024 ** 3})
            qmp.command('migrate', {'uri': 'file:' + str(part)})
            migration_started = True
            deadline = time.monotonic() + SAVE_TIMEOUT
            while True:
                migration = qmp.command('query-migrate')
                phase = migration.get('status')
                if phase == 'completed':
                    break
                if phase in ('failed', 'cancelled'):
                    raise RuntimeError('Machine save failed: ' + migration.get('error-desc', phase))
                if time.monotonic() > deadline:
                    raise RuntimeError('Machine save timed out.')
                time.sleep(.2)
            # Migration completion includes block-device inactivation/flush. Commit only
            # after the complete image and manifest are durable on this filesystem.
            with part.open('rb') as image:
                os.fsync(image.fileno())
            os.replace(part, run / name)
            sync_directory(run)
            image = run / name
            manifest = dict(format=FORMAT, phase='saved', image=name,
                            image_size=image.stat().st_size, image_sha256=digest(image),
                            qemu_sha256=digest(args[0]), command=args,
                            source_pid=source_pid, source_identity=process_identity(source_pid), saved_at=time.time(),
                            disks=[identity(path) for path in drive_paths(args)])
            atomic_json(run / 'saved-machine.json', manifest)
            committed = True
            # No guest shutdown or reset: terminate only after the saved state is durable.
            try:
                qmp.command('quit')
            except (EOFError, OSError, RuntimeError) as error:
                # Quit may close the transport before its acknowledgement arrives.
                # The durable image already exists; wait for the bound source to exit.
                manifest['quit_transport_note'] = str(error)
            deadline = time.monotonic() + 15
            while manifest['source_identity'] is not None and process_identity(source_pid) == manifest['source_identity']:
                if time.monotonic() > deadline:
                    raise RuntimeError('State is saved, but source QEMU has not exited.')
                time.sleep(.05)
            # Closing a qcow2 can update its header timestamp after inactivation.
            manifest['disks'] = [identity(path) for path in drive_paths(args)]
            atomic_json(run / 'saved-machine.json', manifest)
            notify('macOS saved to disk. Open the shortcut to restore this session.')
            return manifest
        except BaseException:
            if not committed:
                if migration_started:
                    try:
                        qmp.command('migrate_cancel')
                        deadline = time.monotonic() + 10
                        while qmp.command('query-migrate').get('status') in ('active', 'setup', 'cancelling'):
                            if time.monotonic() > deadline:
                                raise RuntimeError('Migration cancellation did not finish; guest remains paused.')
                            time.sleep(.1)
                    except Exception:
                        # Do not resume while the migration writer could still read RAM.
                        raise
                qmp.command('cont')
                part.unlink(missing_ok=True)
                if image is not None:
                    image.unlink(missing_ok=True)
            raise


def mark_restoring(run, pid):
    manifest = read_saved(run)
    if not manifest or manifest['phase'] not in ('saved', 'restoring'):
        raise RuntimeError('No saved machine to restore.')
    manifest['phase'] = 'restoring'
    manifest['restore_pid'] = pid
    atomic_json(run / 'saved-machine.json', manifest)


def mark_restored(run):
    manifest = read_saved(run)
    if not manifest or manifest['phase'] not in ('saved', 'restoring'):
        raise RuntimeError('No saved machine to mark restored.')
    manifest['phase'] = 'restored'
    manifest['restored_at'] = time.time()
    atomic_json(run / 'saved-machine.json', manifest)
