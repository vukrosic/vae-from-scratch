#!/bin/sh
# Runs every step in order. Trained networks are kept in nets/, so a second run is fast.
set -e
cd "$(dirname "$0")"
python3 tests.py
for f in step1_squeeze step2_map step3_draw step4_dice step5_cloud step6_pull step7_dial step8_more; do
  echo; echo "== $f"; python3 $f.py
done
