#!/bin/bash
# GPU 0 (now free): confirm solar at full 512-res + a deep-ensemble member.
cd /home/spant/UMich/umich-hackathon/rgb2thermal || exit 1
PY=../weather_experiments/.venv/bin/python
export CUDA_VISIBLE_DEVICES=0
run(){ name="$1"; shift; echo "START $name $(date)" >> logs/progress.md; $PY -u "$@" > "logs/$name.log" 2>&1; echo "END $name $(date) rc=$?" >> logs/progress.md; }
run a1_solar_512  train_a1.py --name a1_solar_512  --use_depth 0 --solar 1 --res 512 --epochs 80 --bs 6
run a1_rgb_s2_512 train_a1.py --name a1_rgb_s2_512 --use_depth 0           --res 512 --epochs 80 --bs 6
echo "SUITE4 DONE $(date)" >> logs/progress.md
