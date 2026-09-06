#!/usr/bin/env bash
# Temporary dedicated C4 guest tuning; record failures for managed interrupts.
# Call only between formal runs, as root. CPU1 stays online as an idle sibling.
set -euo pipefail
out=$1
mkdir -p "$out"
cat /proc/interrupts > "$out/interrupts-before.txt"
if systemctl is-active --quiet irqbalance.service; then
  systemctl stop irqbalance.service
  echo stopped > "$out/irqbalance.txt"
fi
shopt -s nullglob
for file in /sys/devices/virtual/workqueue/cpumask /sys/devices/virtual/workqueue/*/cpumask /proc/irq/default_smp_affinity; do
  [[ -e "$file" ]] || continue
  old=$(cat "$file")
  if printf '1\n' > "$file" 2>> "$out/unmodifiable.txt"; then
    printf '%s: %s -> %s\n' "$file" "$old" "$(cat "$file")" >> "$out/affinity.txt"
  else
    printf '%s: unchanged %s\n' "$file" "$old" >> "$out/affinity.txt"
  fi
done
for file in /proc/irq/*/smp_affinity_list; do
  old=$(cat "$file")
  if printf '0\n' > "$file" 2>> "$out/unmodifiable.txt"; then
    printf '%s: %s -> %s\n' "$file" "$old" "$(cat "$file")" >> "$out/affinity.txt"
  else
    printf '%s: unchanged %s\n' "$file" "$old" >> "$out/affinity.txt"
  fi
done
cat /proc/interrupts > "$out/interrupts-after.txt"
