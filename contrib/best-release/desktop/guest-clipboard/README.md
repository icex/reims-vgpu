The host `guest-clipboard.py` sends only these six source dependencies and
`install.py` over the existing private SSH forward, then runs the installer as
the logged-in guest user. No pasteboard contents are sent over SSH. The host
helper waits at most ten minutes for Aqua and retries installation at most
three times; its own SSH child is terminated when its session owner stops it.

The guest installer preserves the old `org.raphaelgpu.clipboard` files and
records original loaded/disabled states in mode0600
`~/Library/Application Support/Reims/clipboard-agent/migration.json`. It disables
that legacy label persistently to keep the virtio-serial tty exclusive. Any
previous `org.reims.clipboard.plist` is copied alongside the migration state.
The dedicated Reims job runs for the guest lifetime (`--seconds 0`), uses scale1
without a display/capture holder, and logs metadata only. It receives clipboard
text solely through SPICE. Sources are content-compared and AppKit is rebuilt
only when its source hash changes; an unchanged, loaded job is not restarted.
Only an AppKit `read -1` baseline is used for warm-up, without reading old text.

A single timeout, SIGALRM or unavailable-helper return4 no longer permanently
disables text sharing. It invalidates pending mutations, retains an old request
until its stale reply drains, releases the old offer, and retries after one
second. A request that cannot be fulfilled gets a type-NONE response so the
peer's timer can drain. Three consecutive actual helper failures disable text
sharing; a successful helper call resets that count. Ambiguous SPICE response
timeouts remain fail-closed until reconnect. The original bounded text limit
and ownership checks remain. No Apple binary is included; AppKit helper source
is compiled in the guest.

Removal, inside macOS as the same user (first disable the host `clipboard`
setting so bootstrap won't run again):

```sh
/usr/bin/python3 - <<'PY'
import json,os,pathlib,subprocess
root=pathlib.Path.home()/'Library/Application Support/Reims/clipboard-agent'
domain='gui/'+str(os.getuid());state=json.loads((root/'migration.json').read_text())
run=lambda args:subprocess.run(['launchctl']+args,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
run(['bootout',domain+'/org.reims.clipboard'])
plist=pathlib.Path.home()/'Library/LaunchAgents/org.reims.clipboard.plist'
if state['reims_plist_present']:
 plist.write_bytes((root/'previous-org.reims.clipboard.plist').read_bytes());plist.chmod(0o600)
else:plist.unlink(missing_ok=True)
for label,prefix in [('org.reims.clipboard','reims'),('org.raphaelgpu.clipboard','legacy')]:
 run(['disable' if state[prefix+'_disabled'] else 'enable',domain+'/'+label])
 if state[prefix+'_loaded']:
  run(['bootstrap',domain,str(pathlib.Path.home()/'Library/LaunchAgents'/(label+'.plist'))])
print('Previous clipboard job states restored; keep migration.json until verified.')
PY
```

After verifying the prior job, remove only the new
`~/Library/Application Support/Reims/clipboard-agent/` directory. The existing
RaphaelGPU scripts, logs and plist have not been edited. These removal steps do
not change the old home-share agent or the host Raphael service.
