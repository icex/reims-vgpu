"""Restore ordering through the actual shortcut; no QEMU, service or GPU calls."""
import runpy,struct,unittest,tempfile
from pathlib import Path
from unittest.mock import patch,Mock

P=Path(__file__).resolve().parent
api=runpy.run_path(str(P/'reims-macos'))
restore=api['restore_loaded_session']
g=restore.__globals__

class QMP:
    def __init__(self,events):self.events=events;self.state='paused';self.connection=Mock();self.connection.getsockopt.return_value=struct.pack('3i',123,1000,1000)
    def __enter__(self):return self
    def __exit__(self,*_):pass
    def status(self):return self.state
    def command(self,name,arguments=None,timeout=None):
        assert self.events==['restore marker durable'];assert name=='cont';assert timeout==120
        self.events.append('cont');self.state='running'

class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.run=Path(self.tmp.name)
    def test_active_migration_cannot_resume(self):
        for status in ('active','setup','cancelling','pre-switchover','postcopy-active'):
            q=Mock();q.command.return_value={'status':status}
            with self.assertRaises(RuntimeError):api['ensure_not_migrating'](q)
    def test_terminal_migration_allows_source_recovery(self):
        for status in ('completed','failed','cancelled',None):
            q=Mock();q.command.return_value={'status':status};api['ensure_not_migrating'](q)
    def test_marker_is_durable_before_guest_execution(self):
        events=[];q=QMP(events)
        saved={'mark_restoring':lambda run,pid:events.append('restore marker durable'),
               'mark_restored':lambda run:events.append('restored')}
        with patch.dict(g,QMP=lambda run:q,notify=lambda text:None),patch.object(g['runpy'],'run_path',return_value={'window':lambda pid,action:events.append('window')}):
            restore(self.run,saved)
        self.assertEqual(events,['restore marker durable','cont','restored','window'])
    def test_marker_failure_leaves_guest_stopped(self):
        events=[];q=QMP(events)
        def fail(*_):raise OSError('cannot commit restore marker')
        with patch.dict(g,QMP=lambda run:q),self.assertRaises(OSError):
            restore(self.run,{'mark_restoring':fail})
        self.assertEqual(q.state,'paused');self.assertFalse(events)

if __name__=='__main__':unittest.main(verbosity=2)
