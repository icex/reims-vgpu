#!/usr/bin/env python3
"""Copy opt-in SDL drops to macOS Desktop, never interpreting dropped paths as code."""
import argparse,json,os,queue,shlex,signal,socket,stat,subprocess,tarfile,threading,time
from pathlib import Path

# The SSH command is fixed code. Names and bytes travel only in the tar stream.
REMOTE = r'''
import os,sys,tarfile,tempfile,shutil,pathlib,json,subprocess
root=os.path.expanduser('~/Desktop')
stage=tempfile.mkdtemp(prefix='.reims-drop-',dir=root)
try:
 with tarfile.open(fileobj=sys.stdin.buffer,mode='r|*') as archive:
  top=None
  for member in archive:
   parts=pathlib.PurePosixPath(member.name).parts
   if not parts or member.name.startswith('/') or '..' in parts:raise ValueError('unsafe archive name')
   if top is None:top=parts[0]
   if parts[0]!=top:raise ValueError('multiple archive roots')
   dest=os.path.join(stage,*parts)
   parent=os.path.dirname(dest)
   if os.path.commonpath([stage,os.path.realpath(parent)])!=stage:raise ValueError('archive traverses a symlink')
   probe=stage
   for part in parts[:-1]:
    probe=os.path.join(probe,part)
    if os.path.islink(probe):raise ValueError('archive parent is a symlink')
    os.makedirs(probe,exist_ok=True)
   if member.isdir():
    if os.path.islink(dest):raise ValueError('archive directory is a symlink')
    os.makedirs(dest,exist_ok=True)
   elif member.isfile():
    with archive.extractfile(member) as src,open(dest,'xb') as out:shutil.copyfileobj(src,out,1024*1024)
    os.chmod(dest,member.mode & 0o777)
   elif member.issym():os.symlink(member.linkname,dest)
   elif member.islnk():
    link=os.path.realpath(os.path.join(stage,member.linkname))
    if os.path.commonpath([stage,link])!=stage:raise ValueError('unsafe archive hardlink')
    os.link(link,dest)
   else:raise ValueError('special file refused')
 if top is None:raise ValueError('empty archive')
 name=top;stem,suffix=os.path.splitext(top);i=2
 while os.path.lexists(os.path.join(root,name)):
  name=stem+' ('+str(i)+')'+suffix;i+=1
 # macOS renamex_np(RENAME_EXCL) atomically refuses a concurrently created destination.
 import ctypes,errno
 libc=ctypes.CDLL(None,use_errno=True)
 while libc.renamex_np(os.fsencode(os.path.join(stage,top)),os.fsencode(os.path.join(root,name)),4)!=0:
  e=ctypes.get_errno()
  if e!=errno.EEXIST:raise OSError(e,os.strerror(e))
  name=stem+' ('+str(i)+')'+suffix;i+=1
 subprocess.run(['/usr/bin/osascript','-e','on run argv','-e','tell application "Finder" to reveal POSIX file (item 1 of argv)','-e','end run',os.path.join(root,name)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=15)
 print(json.dumps({'name':name,'ok':True}),flush=True)
finally:shutil.rmtree(stage,ignore_errors=True)
'''

def notify(message):
 try:subprocess.run(['notify-send','--app-name=Reims','Reims file drop',message],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=4)
 except (OSError,subprocess.TimeoutExpired):pass

def path_from_packet(packet):
 if not packet or len(packet)>4096 or b'\0' in packet:raise ValueError('invalid drop length or NUL')
 path=Path(os.fsdecode(packet))
 if not path.is_absolute():raise ValueError('drop must be an absolute path')
 info=path.lstat()
 if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)):raise ValueError('special file refused')
 return path

def archive_filter(member):
 if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):raise ValueError('special file in dropped folder refused')
 return member

def ssh_command(port,key,known_hosts):
 return ['ssh','-T','-i',str(key),'-p',str(port),'-o','BatchMode=yes','-o','ConnectTimeout=8','-o','ControlMaster=no','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+str(known_hosts),'bogdan@127.0.0.1','/usr/bin/python3 -c '+shlex.quote(REMOTE)]

def transfer(path,command,stop,children):
 if stop.is_set():return
 notify('Copying '+path.name+' to the macOS Desktop…')
 child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 children.add(child)
 try:
  with tarfile.open(fileobj=child.stdin,mode='w|',format=tarfile.PAX_FORMAT,dereference=False) as archive:
   archive.add(path,arcname=path.name,recursive=True,filter=archive_filter)
  child.stdin.close();child.stdin=None
  out,err=child.communicate(timeout=3600)
  if child.returncode:raise RuntimeError('guest copy failed (SSH exit '+str(child.returncode)+')')
  result=json.loads(out);assert result['ok']
  notify('Copied '+result['name']+' to the macOS Desktop')
  print(json.dumps({'event':'complete','name':result['name']}),flush=True)
 except BaseException:
  if child.poll() is None:child.terminate()
  try:child.wait(timeout=5)
  except subprocess.TimeoutExpired:child.kill();child.wait()
  raise
 finally:children.discard(child)

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--ssh-port',type=int,required=True);parser.add_argument('--ssh-key',type=Path,required=True);parser.add_argument('--known-hosts',type=Path,required=True)
 args=parser.parse_args();os.umask(0o077);run=args.run.resolve(strict=True);info=run.stat()
 if info.st_uid!=os.getuid() or stat.S_IMODE(info.st_mode)!=0o700:raise ValueError('private run directory required')
 endpoint=run/'drop.sock'
 # A preexisting endpoint is never silently replaced.
 sock=socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM);sock.bind(str(endpoint));endpoint.chmod(0o600);sock.settimeout(.5)
 stop=threading.Event();pending=queue.Queue(maxsize=32);children=set()
 def worker():
  while not stop.is_set():
   try:path=pending.get(timeout=.5)
   except queue.Empty:continue
   try:transfer(path,ssh_command(args.ssh_port,args.ssh_key,args.known_hosts),stop,children)
   except Exception as error:
    print(json.dumps({'event':'failed','error':str(error)}),flush=True);notify('Copy failed: '+str(error))
 def end(*_):
  stop.set()
  for child in list(children):
   if child.poll() is None:child.terminate()
 signal.signal(signal.SIGTERM,end);signal.signal(signal.SIGINT,end)
 thread=threading.Thread(target=worker,daemon=True);thread.start()
 print('drop-helper-ready',flush=True)
 try:
  while not stop.is_set():
   try:
    packet,_,flags,_=sock.recvmsg(4097)
    if flags & socket.MSG_TRUNC:raise ValueError('oversized drop packet')
    path=path_from_packet(packet);pending.put_nowait(path)
   except socket.timeout:continue
   except (ValueError,OSError,queue.Full) as error:notify('Drop refused: '+str(error))
 finally:
  end();sock.close();thread.join(timeout=7);endpoint.unlink(missing_ok=True)
if __name__=='__main__':main()
