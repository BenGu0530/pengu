#!/usr/bin/env bash
# capture.sh — record a sketch's serial stream to a CSV on the Mac.
#
# Generic: it knows nothing about which sketch is running. Every probe and walking sketch
# in this folder logs plain CSV over USB serial, so this is the one recorder for all of
# them.
#
#   bash Arduino/capture.sh              # auto-detect port, timestamped file
#   bash Arduino/capture.sh bench_a.csv  # name the file
#   PORT=/dev/cu.usbmodem1101 bash Arduino/capture.sh
#
# Uses macOS's built-in `screen`: keystrokes go to the board (so 'z', '1'..'9', space
# all work) and everything it prints lands in the log file.
#
#   QUIT:  Ctrl-A  then  k  then  y      <-- not Ctrl-C; Ctrl-C leaves the port locked
#   If the port ever gets stuck:  screen -wipe
set -u

OUT="${1:-imu_$(date +%Y%m%d_%H%M%S).csv}"
case "$OUT" in /*) ;; *) OUT="$PWD/$OUT" ;; esac

if [ -z "${PORT:-}" ]; then
  # Arduino boards show up as cu.usbmodem*; prefer that over anything Bluetooth.
  PORT="$(ls /dev/cu.usbmodem* 2>/dev/null | head -1)"
fi
if [ -z "${PORT:-}" ] || [ ! -e "$PORT" ]; then
  echo "No /dev/cu.usbmodem* found. Plug the board in, or set PORT=... explicitly."
  echo "Ports currently present:"; ls /dev/cu.* 2>/dev/null | sed 's/^/  /'
  exit 1
fi

echo "port : $PORT"
echo "file : $OUT"
echo
echo "keys : z = 2s static average (use this for each bench pose)"
echo "       1..9 = label the pose column     space = pause/resume     h = header"
echo "QUIT : Ctrl-A  then  k  then  y"
echo
sleep 1

# macOS ships GNU screen 4.00.03, which has no -Logfile: its -L writes screenlog.0 into
# the current directory. Newer screen (brew) takes -Logfile. Support both.
if screen --version 2>&1 | grep -qE "4\.0[01]"; then
  cd "$(dirname "$OUT")" || exit 1
  rm -f screenlog.0
  screen -L "$PORT" 115200
  if [ -f screenlog.0 ]; then mv -f screenlog.0 "$OUT"; echo "saved: $OUT"; else echo "no screenlog.0 written"; fi
else
  screen -L -Logfile "$OUT" "$PORT" 115200
  echo "saved: $OUT"
fi
