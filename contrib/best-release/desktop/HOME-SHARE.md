# Reims home share

The optional home-share.py companion runs for exactly one run-session.py child
lifetime. Start it only when desktop-settings.json home_share is true (default
false for older settings). Run it beside QEMU, never as a prerequisite for QEMU startup. Stop it on
all session exits and allow 90 seconds for cooperative Docker cleanup. A missing
Docker daemon, image, or guest SSH must only produce a log/desktop notification.

CLI: home-share.py --run RUN --ssh-port PORT --ssh-key KEY --known-hosts FILE.
The default state is ~/.config/reims-macos/home-share, mode0700. The new password
is a random256-bit hex string in password, mode0600. Samba receives it through a
mounted private file and smbpasswd stdin. Guest setup transmits it only on SSH
stdin into security's interactive stdin; no password appears in process argv,
logs, packaged scripts, or source. The guest stores the internet-password item in
its login keychain, scoped to10.0.2.2,port4450,pathReimsHome,protocolSMB.

The local image raphael-macos-shares:local supplies Samba and its existing
macosshare UNIX account (checked UID/GID1000). This new service uses its own
password database in container tmpfs, regenerated from the private credential on
every start. No root-owned Samba state is written to the host. It does not touch the older Raphael container/service/state.
The container uses host networking, with smbd bound exclusively to127.0.0.1:4450;
there is no published Docker bridge address and NetBIOS is disabled. Client URL:
smb://10.0.2.2:4450/ReimsHome. All share operations run as UID/GID1000.

The complete home is writable, including dotfiles (Finder Cmd+Shift+period).
This gives macOS programs access to ~/.ssh and other host credentials. Protected
storage is excluded: every component named reims-9070xt, plus exact basenames of
existing disk images under macos-vm, all active VM drive paths, and the disposable pristine snapshot drives. Files named macos.qcow2 are always vetoed. Recognized image suffixes are qcow2,qcow,img,
raw,vmdk,vdi,dmg,iso. Samba matches components rather than relative paths, so any
duplicate image basename elsewhere in home is also excluded. The start receipt
lists every protected image path and excluded basename. Other macos-vm contents
remain writable. Defense against symlink aliases additionally mounts an empty,
read-only directory over reims-9070xt and empty read-only files over each real
image inside the container; host files are never modified. Symlinks within the
share are followed; escapes from the exported home are refused.

.DS_Store and ._* are vetoed with delete-veto-files enabled. fruit and
streams_xattr store Mac metadata in xattrs; the guest disables network DS_Store
creation, and Samba spotlight=no provides no network indexing service.
SMB3 and mandatory signing are enabled. Encryption defaults to if_required
rather than required because traffic stays in host loopback and the guest's
private slirp link; clients can request encryption. The optional --encryption
required permits a measured encrypted comparison. A copy-speed result must name
the chosen policy and may not be described as a desktop/GPU speedup.

Idempotent SSH setup waits for Aqua/Finder. It archives and unloads the obsolete
org.raphael.linux-shares LaunchAgent without deleting any of its keychain items;
installs org.reims.home-share with15-second retry/remount checks; and adds the
Desktop/Linux Home link to /Volumes/ReimsHome. Password retrieval and mount are
inside AppleScript memory, with subprocess output suppressed.

No-hardware tests: python3 -B test_home_share.py. Real guest mount/reboot/copy,
login-keychain access, visibility, throughput and lifetime cleanup are separate
runtime gates.

## Removal

While no VM is running, set desktop-settings.json home_share=false. In macOS:

```sh
launchctl bootout gui/$(id -u)/org.reims.home-share
rm "$HOME/Library/LaunchAgents/org.reims.home-share.plist"
umount /Volumes/ReimsHome
rm "$HOME/Desktop/Linux Home"
security delete-internet-password -a macosshare -s 10.0.2.2 -p ReimsHome -P 4450 -r 'smb '
```

The guest's Library/Application Support/Reims/home-share directory contains the
original DSDontWriteNetworkStores value (network-stores.before; empty means
absent), the original obsolete agent (.before) and disabled agent (.disabled).
Restore the old preference or delete its key when originally absent. Restore the
old agent only if deliberately returning to the old Raphael shares. Remove the
Reims support directory after inspecting those backups. The host config/state
can be removed after all Reims-owned home-share containers are absent. No host
package or system configuration was installed.
