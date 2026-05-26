#!/bin/bash
# Waits for suite1 (A1/A2) to release GPU 0, then runs A4 physics + a second variant.
cd /home/spant/UMich/umich-hackathon/rgb2thermal || exit 1
PY=../weather_experiments/.venv/bin/python
export CUDA_VISIBLE_DEVICES=0
while pgrep -f "train_a[12].py" >/dev/null; do sleep 60; done
echo "SUITE2 START $(date)" >> logs/progress.md
run(){ name="$1"; shift; echo "START $name $(date)" >> logs/progress.md; $PY -u "$@" > "logs/$name.log" 2>&1; echo "END   $name $(date) rc=$?" >> logs/progress.md; }
run a4_physics  train_a4.py --name a4_physics  --encoder convnext_tiny --use_depth 1 --K 6  --epochs 80 --bs 6
run a4_physics8 train_a4.py --name a4_physics8 --encoder convnext_tiny --use_depth 1 --K 8  --epochs 80 --bs 6
echo "SUITE2 DONE $(date)" >> logs/progress.md
