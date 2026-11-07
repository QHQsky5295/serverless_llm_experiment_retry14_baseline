#!/usr/bin/env bash
# D80 monitor_remote_services_v2.sh, restricted to one explicitly owned invocation.
# Run over SSH stdin; no shared service configuration or other model is changed.
set -euo pipefail
unit=${1:?artifact unit required}
invocation=${2:?owned invocation required}
case "$unit" in primelora-artifact-tc-7b.service|primelora-artifact-tc-3b.service) ;; *) exit 2 ;; esac
[[ "$invocation" =~ ^[a-f0-9]{32}$ ]]
base=/home/lab14/primelora_remote/tc/d80_20260927
test "$(systemctl --user show "$unit" -p InvocationID --value)" = "$invocation"
cg=/sys/fs/cgroup$(systemctl --user show "$unit" -p ControlGroup --value)
test "$(cat "$cg/memory.high")" = 2147483648
test "$(cat "$cg/memory.max")" = 4294967296
test "$(cat "$cg/memory.swap.max")" = 0
systemctl --user show "$unit" -p Id -p InvocationID -p MainPID -p ControlGroup -p MemoryHigh -p MemoryMax -p MemorySwapMax -p TasksMax
owned_stop() {
  if test "$(systemctl --user show "$unit" -p InvocationID --value)" = "$invocation"; then
    systemctl --user stop "$unit"
  fi
}
trap owned_stop EXIT
trap 'exit 0' TERM INT
high_pressure=0
sample=0
while test "$(systemctl --user show "$unit" -p InvocationID --value)" = "$invocation" &&
      test "$(systemctl --user show "$unit" -p MainPID --value)" != 0; do
  sample=$((sample+1))
  available_kib=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
  pressure=$(awk '/^full /{split($2,a,"=");print a[2]}' /proc/pressure/memory)
  if test "$available_kib" -lt 25165824 && awk "BEGIN{exit !($pressure>=10)}"; then
    high_pressure=$((high_pressure+1))
  else high_pressure=0; fi
  if test "$available_kib" -lt 16777216 || test "$high_pressure" -ge 10; then
    printf 'safety_abort sample=%s mem_available_kib=%s full_avg10=%s\n' "$sample" "$available_kib" "$pressure"
    exit 70
  fi
  if test $((sample%30)) -eq 1; then
    free=$(df -B1 --output=avail "$base" | tail -1 | tr -d ' ')
    test "$free" -ge 17179869184 || exit 71
    test "$(df --output=iavail "$base" | tail -1 | tr -d ' ')" -ge 1000 || exit 72
  fi
  printf 'sample=%s utc=%s mem_available_kib=%s full_avg10=%s disk_free_bytes=%s\n' "$sample" "$(date -u +%FT%TZ)" "$available_kib" "$pressure" "$free"
  sed 's/^/host_net /' /proc/net/dev
  sed 's/^/host_disk /' /proc/diskstats
  sed -n '1s/^/host_cpu /p' /proc/stat
  for name in memory.current memory.peak memory.swap.current pids.current memory.events cpu.stat memory.stat memory.pressure; do
    if test -r "$cg/$name"; then sed "s/^/$unit $name /" "$cg/$name"; fi
  done
  if test -r "$cg/io.stat"; then
    sed "s/^/$unit io /" "$cg/io.stat"
  else printf 'unit=%s io_accounting=not_delegated\n' "$unit"; fi
  sleep 1
done
