# Guest clock after saved-session restore

Install in macOS with administrator approval using `sudo ./install.sh`.
The root-owned launch daemon checks Apple's NTP server every 15 seconds,
including after a saved session resumes. It steps only offsets over two
seconds; normal small adjustments remain with macOS timed. Offline guests
retry at the next interval. No credentials or host clock are accepted.

Uninstall with `sudo launchctl bootout system/io.github.icex.reims-clock`,
then remove `/Library/LaunchDaemons/io.github.icex.reims-clock.plist` and
`/usr/local/libexec/reims-sync-clock` with administrator approval.
