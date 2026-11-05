import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

HERE=Path(__file__).resolve().parent

def load(name):
    spec=importlib.util.spec_from_file_location(name.replace('-','_'),HERE/(name+'.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

control=load('sleep-control')
helper=load('close-guest')

class Guest:
    def __init__(self,state):self.state=state;self.commands=[];self.connection=MagicMock();self.connection.getsockopt.return_value=__import__('struct').pack('3i',123,1000,1000)
    def __enter__(self):return self
    def __exit__(self,*_):pass
    def status(self):return self.state
    def command(self,c):
        self.commands.append(c)
        self.state={'stop':'paused','cont':'running','system_wakeup':'running'}[c]

class Tests(unittest.TestCase):
    def transition(self,state,action,expected):
        guest=Guest(state)
        with tempfile.TemporaryDirectory() as d,patch.object(control.runpy,'run_path',return_value={'QMP':lambda _:guest}),patch.object(control,'window',return_value='ok') as win:
            result=control.transition(Path(d),action)
            self.assertEqual(guest.commands,expected)
            self.assertEqual(result['after'],'paused' if action=='pause' else 'running')
            self.assertEqual(win.call_args.args,(123,'minimize' if action=='pause' else 'show'))
    def test_pause(self):self.transition('running','pause',['stop'])
    def test_repeated_close(self):self.transition('paused','pause',[])
    def test_resume(self):self.transition('paused','resume',['cont'])
    def test_already_running_restores(self):self.transition('running','resume',[])
    def test_s3_wake(self):self.transition('suspended','resume',['system_wakeup'])
    def test_invalid_state_retained(self):
        guest=Guest('shutdown')
        with tempfile.TemporaryDirectory() as d,patch.object(control.runpy,'run_path',return_value={'QMP':lambda _:guest}):
            with self.assertRaises(RuntimeError):control.transition(Path(d),'pause')
        self.assertEqual(guest.commands,[])
    def test_compositor_failure_retains_guest(self):
        guest=Guest('running')
        with tempfile.TemporaryDirectory() as d,patch.object(control.runpy,'run_path',return_value={'QMP':lambda _:guest}),patch.object(control,'window',side_effect=RuntimeError('KWin unavailable')):
            result=control.transition(Path(d),'pause');self.assertIn('window_error',result)
        self.assertEqual(guest.state,'paused')
    def host(self,initial):
        host=helper.HostSleep.__new__(helper.HostSleep);host.run=Path('/unused');host.resume_after_host=False
        host.release=MagicMock();host.inhibit=MagicMock()
        with patch.object(helper.control,'transition',side_effect=[{'before':initial,'after':'paused'},{'before':'paused','after':'running'}]) as action,patch.object(helper.legacy,'record'):
            host.prepare(True);host.release.assert_called_once()
            host.prepare(False);host.inhibit.assert_called_once()
            self.assertEqual(action.call_count,2 if initial=='running' else 1)
            self.assertFalse(host.resume_after_host)
    def test_host_sleep_running(self):self.host('running')
    def test_host_sleep_preserves_x_pause(self):self.host('paused')
    def test_sleep_failure_releases_delay(self):
        host=helper.HostSleep.__new__(helper.HostSleep);host.run=Path('/unused');host.release=MagicMock()
        with patch.object(helper.control,'transition',side_effect=RuntimeError('QMP offline')),patch.object(helper.legacy,'notify'),patch.object(helper.legacy,'record'):
            host.prepare(True);host.release.assert_called_once()

if __name__=='__main__':unittest.main()
