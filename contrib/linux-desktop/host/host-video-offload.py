#!/usr/bin/env python3
"""Manual SSH-based VAAPI prototype. This does not register a macOS VideoToolbox codec."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys
import time
import uuid

REMOTE='''from pathlib import Path
import sys,shutil,os
os.umask(0o077)
root=Path.home()/'Library/Application Support/Reims/video-tests'
root.mkdir(parents=True,exist_ok=True)
name=sys.argv[2]
if Path(name).name!=name or name in ('.','..'):raise SystemExit('Filename required')
p=root/name
if sys.argv[1]=='read':
 if p.is_symlink() or not p.is_file() or p.stat().st_size>1024**3:raise SystemExit('Invalid input')
 with p.open('rb')as f:shutil.copyfileobj(f,sys.stdout.buffer)
elif sys.argv[1]=='write':
 with p.open('xb')as f:shutil.copyfileobj(sys.stdin.buffer,f)
 p.chmod(0o600)
elif sys.argv[1]=='commit':
 out=root/sys.argv[3]
 if Path(sys.argv[3]).name!=sys.argv[3] or out.exists():raise SystemExit('Output exists or is invalid')
 p.replace(out)
elif sys.argv[1]=='remove':p.unlink(missing_ok=True)
'''

def remote(ssh,action,*names):
 return ssh+['/usr/bin/python3 -c '+shlex.quote(REMOTE)+' '+shlex.join([action,*names])]

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('input',help='filename in the Reims guest video-tests directory')
 parser.add_argument('output',help='new MP4 filename in the same directory')
 parser.add_argument('--codec',choices=('h264','hevc'),default='hevc')
 parser.add_argument('--ssh-host',required=True,help='guest alias configured in ~/.ssh/config; host key must already be trusted')
 parser.add_argument('--render-node',default='/dev/dri/renderD128')
 args=parser.parse_args()
 if args.ssh_host.startswith('-') or any(c.isspace() for c in args.ssh_host):
  parser.error('Use one SSH host alias')
 ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',
      '-o','StrictHostKeyChecking=yes',args.ssh_host]
 if any(Path(n).name!=n or n in ('.','..') for n in (args.input,args.output)):
  parser.error('Use filenames only inside the guest video-tests directory')
 temporary='offload-'+uuid.uuid4().hex+'.partial'
 start=time.monotonic();processes=[]
 try:
  source=subprocess.Popen(remote(ssh,'read',args.input),stdin=subprocess.DEVNULL,stdout=subprocess.PIPE);processes.append(source)
  encoder=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','info','-hwaccel','vaapi',
      '-hwaccel_device',args.render_node,'-hwaccel_output_format','vaapi','-i','pipe:0',
      '-map','0:v:0','-an','-c:v',args.codec+'_vaapi','-b:v','63000000',
      '-movflags','frag_keyframe+empty_moov+default_base_moof','-f','mp4','pipe:1'],
      stdin=source.stdout,stdout=subprocess.PIPE);processes.append(encoder);source.stdout.close()
  sink=subprocess.Popen(remote(ssh,'write',temporary),stdin=encoder.stdout);processes.append(sink);encoder.stdout.close()
  codes=[p.wait(timeout=180)for p in processes]
  if any(codes):raise RuntimeError('Offload failed: '+str(codes))
  subprocess.run(remote(ssh,'commit',temporary,args.output),stdin=subprocess.DEVNULL,check=True,timeout=15)
  print(f'Host VAAPI {args.codec} prototype completed in {time.monotonic()-start:.3f}s; guest file {args.output}')
 finally:
  # Cleanup reports its own timeouts instead of raising, so it cannot mask the original error.
  for p in processes:
   if p.poll()is None:
    p.terminate()
    try:p.wait(timeout=10)
    except subprocess.TimeoutExpired:p.kill();p.wait()
  try:
   subprocess.run(remote(ssh,'remove',temporary),stdin=subprocess.DEVNULL,
                  stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=15)
  except subprocess.TimeoutExpired:
   print(f'Could not remove guest temporary {temporary}: ssh timed out',file=sys.stderr)

if __name__=='__main__':main()
