#!/bin/bash
# Sequential training suite on GPU 0 (unattended). Each run logs to logs/<name>.log.
cd /home/spant/UMich/umich-hackathon/rgb2thermal || exit 1
PY=../weather_experiments/.venv/bin/python
export CUDA_VISIBLE_DEVICES=0
mkdir -p logs checkpoints
echo "SUITE START $(date)" > logs/progress.md
run(){ name="$1"; shift; echo "START $name $(date)" >> logs/progress.md; $PY -u "$@" > "logs/$name.log" 2>&1; echo "END   $name $(date) rc=$?" >> logs/progress.md; }

run a1_rgb     train_a1.py --name a1_rgb     --encoder convnext_tiny  --use_depth 0              --epochs 80  --bs 6
run a1_rgbd    train_a1.py --name a1_rgbd    --encoder convnext_tiny  --use_depth 1              --epochs 80  --bs 6
run a1_rgbda   train_a1.py --name a1_rgbda   --encoder convnext_tiny  --use_depth 1 --use_alpha 1 --epochs 80 --bs 6
run a1_unreg   train_a1.py --name a1_unreg   --encoder convnext_tiny  --use_depth 0 --unreg 1     --epochs 80  --bs 6
run a1_small   train_a1.py --name a1_small   --encoder convnext_small --use_depth 1              --epochs 70  --bs 5
run a2_gan     train_a2.py --name a2_gan     --encoder convnext_tiny  --use_depth 1              --epochs 100 --bs 5

echo "SUITE DONE $(date)" >> logs/progress.md
