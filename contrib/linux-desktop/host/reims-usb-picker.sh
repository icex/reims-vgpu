#!/usr/bin/env bash
# Desktop USB picker for the macOS guest: one checklist of host USB devices; checked ones are
# redirected to macOS (up to two, the slots of reims-usb.sh beside this script), unchecked
# ones are given back. Root hubs and USB hubs are not offered. A VID:PID shared by more than
# one attached device is not offered unless already redirected, since reims-usb.sh cannot
# tell them apart. Messages go to a desktop notification.
set -uo pipefail
USB="$(dirname "$(readlink -f "$0")")/reims-usb.sh"
TITLE="macOS (Reims)"
notify() { notify-send -a "${TITLE}" -i drive-removable-media "${TITLE}" "$1" 2>/dev/null || echo "$1"; }

declare -A attached=()      # VID:PID -> slot
while read -r slot id; do
    [[ "${id}" =~ ^[0-9a-fA-F]{4}:[0-9a-fA-F]{4}$ ]] && attached["${id,,}"]="${slot}"
done < <("${USB}" status | sed -n 's/^slot \([12]\): \(.*\)$/\1 \2/p')

declare -A seen=()          # VID:PID -> number of attached devices
while read -r _ _ _ _ _ id _; do
    id="${id,,}"; seen["${id}"]=$(( ${seen[${id}]:-0} + 1 ))
done < <(lsusb)

items=(); ambiguous=()
while read -r _ bus _ number _ id name; do
    id="${id,,}"
    [[ "${id}" == 1d6b:* || "${name}" =~ [Hh]ub$ ]] && continue
    # A device already redirected stays listed so it can be given back (detach is by slot).
    if (( ${seen[${id}]:-0} > 1 )) && [[ -z "${attached[${id}]:-}" ]]; then
        [[ " ${ambiguous[*]} " == *" ${id} "* ]] || ambiguous+=("${id}")
        continue
    fi
    label=$(python3 "$(dirname "${USB}")/usb-device-info.py" "${bus}" "${number%:}" "${name:-${id}}") || label="${name:-${id}}"
    state=off; [[ -n "${attached[${id}]:-}" ]] && state=on
    items+=("${id}" "${label} (${id})" "${state}")
done < <(lsusb)
(( ${#ambiguous[@]} )) && notify "Not offered: more than one attached device has ID ${ambiguous[*]}."
(( ${#items[@]} )) || { notify "No USB devices to pass to macOS."; exit 0; }

if command -v kdialog >/dev/null; then
    chosen=$(kdialog --title "${TITLE}" --separate-output \
        --checklist "Select the USB storage device to mount a drive in macOS (at most two devices):" "${items[@]}") || exit 0
elif command -v zenity >/dev/null; then
    args=(); for ((i=0; i<${#items[@]}; i+=3)); do
        args+=("$([[ ${items[i+2]} == on ]] && echo TRUE || echo FALSE)" "${items[i]}" "${items[i+1]}")
    done
    chosen=$(zenity --list --checklist --title "${TITLE}" --text "Select the USB storage device to mount a drive in macOS (at most two devices):" \
        --column Use --column ID --column Device --separator=$'\n' "${args[@]}") || exit 0
else
    notify "Neither kdialog nor zenity is installed: cannot show the USB picker."
    exit 1
fi
mapfile -t want < <(printf '%s\n' "${chosen}" | sed '/^$/d' | tr 'A-F' 'a-f')
(( ${#want[@]} <= 2 )) || { notify "macOS can use at most two USB devices at a time."; exit 1; }

messages=()
for id in "${!attached[@]}"; do
    printf '%s\n' "${want[@]}" | grep -qx "${id}" && continue
    "${USB}" detach "${attached[${id}]}" >/dev/null && messages+=("Returned ${id} to Linux.")
done
for id in "${want[@]}"; do
    [[ -n "${attached[${id}]:-}" ]] && continue
    if out=$("${USB}" attach "${id}" 2>&1); then messages+=("Connected ${id} to macOS.")
    else messages+=("Could not connect ${id}: ${out##*$'\n'}"); fi
done
(( ${#messages[@]} )) && notify "$(printf '%s\n' "${messages[@]}")"
exit 0
