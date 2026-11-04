import importlib.util,unittest,tempfile,os,tarfile,io,subprocess,json,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('drop',Path(__file__).with_name('file-drop.py'));drop=importlib.util.module_from_spec(spec);spec.loader.exec_module(drop)
class DropTests(unittest.TestCase):
 def test_hostile_names_are_data_and_archive_roundtrips(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)
   for name in ['space name',"single' double\"",'new\nline','$(touch HACKED)','Καλημέρα 文件']:
    path=root/name;path.write_text(name)
    self.assertEqual(drop.path_from_packet(os.fsencode(path)),path)
    data=io.BytesIO()
    with tarfile.open(fileobj=data,mode='w') as archive:archive.add(path,arcname=path.name,filter=drop.archive_filter)
    with tarfile.open(fileobj=io.BytesIO(data.getvalue())) as archive:self.assertEqual(archive.extractfile(name).read(),name.encode())
   self.assertFalse((root/'HACKED').exists())
 def test_limits_and_special_files(self):
  for packet in [b'',b'relative',b'/a\0b',b'/'*4097]:
   with self.assertRaises(ValueError):drop.path_from_packet(packet)
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'fifo';os.mkfifo(path)
   with self.assertRaises(ValueError):drop.path_from_packet(os.fsencode(path))
 def test_archive_symlink_is_not_dereferenced(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);(root/'link').symlink_to('/should-not-be-read');data=io.BytesIO()
   with tarfile.open(fileobj=data,mode='w',dereference=False) as archive:archive.add(root/'link',arcname='link',filter=drop.archive_filter)
   with tarfile.open(fileobj=io.BytesIO(data.getvalue())) as archive:self.assertTrue(archive.getmember('link').issym())
 def test_remote_rejects_traversal_before_writing(self):
  # Linux cannot run macOS renamex_np; traversal fails before that call.
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);(root/'Desktop').mkdir();data=io.BytesIO()
   with tarfile.open(fileobj=data,mode='w') as archive:
    z=tarfile.TarInfo('../escape');z.size=1;archive.addfile(z,io.BytesIO(b'x'))
   r=subprocess.run([sys.executable,'-c',drop.REMOTE],input=data.getvalue(),capture_output=True,env=dict(os.environ,HOME=tmp))
   self.assertNotEqual(r.returncode,0);self.assertFalse((root/'escape').exists());self.assertEqual(list((root/'Desktop').iterdir()),[])
if __name__=='__main__':unittest.main()
