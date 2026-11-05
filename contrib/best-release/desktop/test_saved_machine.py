"""Saved-machine transaction tests with a simulated QMP writer; no VM or GPU."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('saved_machine',P/'saved-machine.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class QMP:
    def __init__(self, run, fail=None):self.run=run;self.calls=[];self.state='paused';self.fail=fail
    def status(self):return self.state
    def command(self, name, args=None):
        self.calls.append(name)
        if name=='migrate':
            self.part=Path(args['uri'].removeprefix('file:'));self.part.write_bytes(b'complete simulated machine image')
        if name=='query-migrate':return {'status':'failed' if self.fail=='migration' else 'completed','error-desc':'injected failure'}
        if name=='cont':self.state='running'
        if name=='quit':
            # Exit may be requested only after durable image + matching durable manifest.
            manifest=m.read_saved(self.run)
            assert manifest['phase']=='saved'
            assert m.digest(m.image_path(self.run,manifest))==manifest['image_sha256']
            if self.fail=='quit':raise EOFError('source exited before ack')
        return {}

class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.run=Path(self.tmp.name)
        self.exe=self.run/'qemu';self.exe.write_bytes(b'exact QEMU build')
        self.disk=self.run/'macos.qcow2';self.disk.write_bytes(b'disk at saved point')
        self.args=[str(self.exe),'-m','16G','-drive','if=none,file='+str(self.disk)]
        (self.run/'command.json').write_text(json.dumps(self.args))
        self.space=patch.object(m.shutil,'disk_usage',return_value=type('Space',(),{'free':100*1024**3})());self.space.start();self.addCleanup(self.space.stop)
    def save(self, qmp=None):return m.save(self.run,qmp or QMP(self.run),99999999)
    def test_success_requires_commit_before_quit(self):
        q=QMP(self.run);r=self.save(q)
        self.assertEqual(q.calls[-1],'quit');self.assertNotIn('cont',q.calls)
        self.assertEqual(m.validate_restore(self.run,self.args),self.run/r['image'])
        self.assertEqual((self.run/'saved-machine.json').stat().st_mode&0o777,0o600)
    def test_migration_failure_keeps_process_and_recovers(self):
        q=QMP(self.run,'migration')
        with self.assertRaisesRegex(RuntimeError,'injected failure'):self.save(q)
        self.assertNotIn('quit',q.calls);self.assertIn('cont',q.calls);self.assertFalse((self.run/'saved-machine.json').exists())
        self.assertFalse(list(self.run.glob('*.partial')))
    def test_manifest_write_failure_never_exits(self):
        q=QMP(self.run)
        with patch.object(m,'atomic_json',side_effect=OSError('disk full')),self.assertRaises(OSError):self.save(q)
        self.assertNotIn('quit',q.calls);self.assertIn('cont',q.calls);self.assertFalse(list(self.run.glob('saved-machine-*.bin')))
    def test_ack_lost_after_commit_preserves_image(self):
        q=QMP(self.run,'quit')
        self.save(q)
        self.assertEqual(m.read_saved(self.run)['phase'],'saved');self.assertNotIn('cont',q.calls)
    def test_changed_disk_refuses_restore(self):
        self.save();self.disk.write_bytes(b'changed guest disk')
        with self.assertRaisesRegex(RuntimeError,'disk changed'):m.validate_restore(self.run,self.args)
    def test_changed_build_refuses_restore(self):
        self.save();self.exe.write_bytes(b'another QEMU build')
        with self.assertRaisesRegex(RuntimeError,'exact QEMU build'):m.validate_restore(self.run,self.args)
    def test_changed_config_refuses_restore(self):
        self.save()
        with self.assertRaisesRegex(RuntimeError,'original QEMU configuration'):m.validate_restore(self.run,self.args+['-smp','4'])
    def test_corrupt_image_refuses_restore(self):
        r=self.save();(self.run/r['image']).write_bytes(b'damaged')
        with self.assertRaisesRegex(RuntimeError,'incomplete or damaged'):m.validate_restore(self.run,self.args)
    def test_duplicate_save_does_not_overwrite_session(self):
        r=self.save();q=QMP(self.run)
        with self.assertRaisesRegex(RuntimeError,'already exists'):self.save(q)
        self.assertFalse(q.calls);self.assertTrue((self.run/r['image']).exists())
    def test_mark_restored_prevents_reuse(self):
        self.save();m.mark_restored(self.run);self.assertIsNone(m.validate_restore(self.run,self.args))
    def test_incomplete_restore_never_falls_back_to_cold_boot(self):
        self.save();m.mark_restoring(self.run,99999998)
        with self.assertRaisesRegex(RuntimeError,'previous restore did not finish'):m.validate_restore(self.run,self.args)
        m.mark_restored(self.run);self.assertIsNone(m.validate_restore(self.run,self.args))
    def test_unknown_phase_is_refused(self):
        r=self.save();r['phase']='damaged';m.atomic_json(self.run/'saved-machine.json',r)
        with self.assertRaisesRegex(RuntimeError,'phase'):m.validate_restore(self.run,self.args)
    def test_no_space_does_not_pause_or_quit(self):
        q=QMP(self.run)
        with patch.object(m.shutil,'disk_usage',return_value=type('Space',(),{'free':0})()),self.assertRaisesRegex(RuntimeError,'free space'):self.save(q)
        self.assertFalse(q.calls)
    def test_image_name_traversal_refused(self):
        r=self.save();r['image']='../saved-machine-evil.bin';m.atomic_json(self.run/'saved-machine.json',r)
        with self.assertRaisesRegex(RuntimeError,'image name'):m.validate_restore(self.run,self.args)

if __name__=='__main__':unittest.main(verbosity=2)
