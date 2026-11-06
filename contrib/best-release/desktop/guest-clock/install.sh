#!/bin/sh
set -eu
[ "$(id -u)" = 0 ] || { echo 'Run with macOS administrator approval.' >&2; exit 1; }
source_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
install -d -o root -g wheel -m 755 /usr/local/libexec
install -o root -g wheel -m 755 "$source_dir/sync-clock.sh" /usr/local/libexec/reims-sync-clock
cat > /Library/LaunchDaemons/io.github.icex.reims-clock.plist <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>Label</key><string>io.github.icex.reims-clock</string>
<key>ProgramArguments</key><array><string>/usr/local/libexec/reims-sync-clock</string></array>
<key>RunAtLoad</key><true/>
<key>StartInterval</key><integer>15</integer>
<key>ProcessType</key><string>Background</string>
</dict></plist>
PLIST
chown root:wheel /Library/LaunchDaemons/io.github.icex.reims-clock.plist
chmod 644 /Library/LaunchDaemons/io.github.icex.reims-clock.plist
plutil -lint /Library/LaunchDaemons/io.github.icex.reims-clock.plist
launchctl bootout system/io.github.icex.reims-clock 2>/dev/null || true
launchctl bootstrap system /Library/LaunchDaemons/io.github.icex.reims-clock.plist
