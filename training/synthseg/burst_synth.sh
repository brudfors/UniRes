#!/bin/bash
# Watchdog burst-loop: resume DDP synth-restore training from last.pt across pod livelocks.
OUT=/workspace/data/synthseg_restore/runs/synth_thick
LOG=$OUT/train.log
TARGET=40000
STALL=400          # seconds of no log progress -> assume livelock, kill & resume
GPUS=0,1,6,7
NPROC=4
PORT=29537
mkdir -p $OUT
cd /workspace/code/UniRes
export PYTHONPATH=/workspace/code/UniRes
while true; do
  st=$(python -c "import torch;print(torch.load('$OUT/last.pt',map_location='cpu').get('step',0))" 2>/dev/null); st=${st:-0}
  [ "$st" -ge "$TARGET" ] && { echo "[wd] TARGET reached at step $st $(date)" >> $LOG; break; }
  echo "[wd] launch from step $st $(date)" >> $LOG
  CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NPROC --master_port=$PORT \
    training/synthseg/train_synth.py --out $OUT --iters $TARGET \
    --batch 8 --patch 96 --sigma_max 0.05 --thick_max 8.0 --blur_max 1.0 \
    --save_every 250 --val_every 1000 --grad_w 0.5 --res_cond 0 >> $LOG 2>&1 &
  PID=$!
  while kill -0 $PID 2>/dev/null; do
    sleep 45
    m=$(stat -c %Y $LOG 2>/dev/null || echo 0)
    [ $(( $(date +%s) - m )) -gt $STALL ] && { echo "[wd] STALL kill $PID $(date)" >> $LOG; pkill -9 -P $PID 2>/dev/null; kill -9 $PID 2>/dev/null; sleep 8; break; }
  done
  sleep 5
done
