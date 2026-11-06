#!/bin/sh
# Saved RAM retains an old wall clock. Only step material drift; leave small
# adjustments to timed. Network failure leaves the clock untouched.
offset=$(/usr/bin/sntp -t 3 time.apple.com 2>/dev/null) || exit 0
printf '%s\n' "$offset" | /usr/bin/awk '
  $1 ~ /^[+-][0-9]+[.][0-9]+$/ && ($1 > 2 || $1 < -2) { drift = 1 }
  END { exit !drift }
' || exit 0
exec /usr/bin/sntp -sS -t 3 time.apple.com
