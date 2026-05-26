#!/bin/bash
# Small-footprint head-to-head on GPU 1 (shared, ~3.6GB free): does solar geometry help?
# a1_rgb2 = control (RGB only), a1_solar = +sun elevation/azimuth. Same settings -> fair val comparison.
cd /home/spant/UMich/umich-hackathon/rgb2thermal || exit 1
PY=../weather_experiments/.venv/bin/python
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run(){ name="$1"; shift; echo "START $name $(date)" >> logs/progress.md; $PY -u "$@" > "logs/$name.log" 2>&1; echo "END $name $(date) rc=$?" >> logs/progress.md; }
run a1_rgb2  train_a1.py --name a1_rgb2  --use_depth 0           --res 320 --bs 2 --epochs 70
run a1_solar train_a1.py --name a1_solar --use_depth 0 --solar 1 --res 320 --bs 2 --epochs 70
mv checkpoints/a1_solar ../rgb2thermal_v2/checkpoints/ 2>/dev/null
echo "SUITE3 DONE $(date)" >> logs/progress.md
