#!/bin/bash
# Watchdog burst-loop for SINGLE-GPU synth-restore training.
# Trains to /dev/shm (tmpfs; NFS /workspace stalls under load -> livelock), and MIRRORS
# last.pt/best.pt to persistent NFS every ~45s so a pod reset never loses hours.
# DDP is avoided on purpose: on this CPU-saturated shared box NCCL allreduce starves and
# collapses throughput to <0.15 it/s, whereas a single process runs at ~1.1 it/s.
GPU=${GPU:-0}
SHM=/dev/shm/synthseg/runs/synth_thick
NFS=/workspace/data/synthseg_restore/runs/synth_thick
LOG=$SHM/train.log
TARGET=${TARGET:-40000}
STALL=${STALL:-600}   # seconds of no log progress -> assume livelock, kill & resume.
#  600s (NOT lower): a slow-but-ALIVE process under heavy box load must not be SIGKILL'd
#  mid-step -- killing at 150-200s just storms (re-imports torch + rebuilds the 34M net
#  every ~1min) and makes ZERO net progress. Better to let a slow process crawl forward.
mkdir -p $SHM $NFS /dev/shm/synthseg/label_maps
cp -n /workspace/data/synthseg_restore/label_maps/*.nii.gz /dev/shm/synthseg/label_maps/ 2>/dev/null
# resume from whichever checkpoint is newer (tmpfs may have been wiped -> fall back to NFS)
if [ ! -f $SHM/last.pt ] && [ -f $NFS/last.pt ]; then cp $NFS/last.pt $SHM/last.pt; cp $NFS/best.pt $SHM/best.pt 2>/dev/null; fi
cd /workspace/code/UniRes
export PYTHONPATH=/workspace/code/UniRes
mirror() { [ -f $SHM/last.pt ] && cp -f $SHM/last.pt $NFS/last.pt 2>/dev/null; [ -f $SHM/best.pt ] && cp -f $SHM/best.pt $NFS/best.pt 2>/dev/null; cp -f $LOG $NFS/train.log 2>/dev/null; }
while true; do
  st=$(python -c "import torch;print(torch.load('$SHM/last.pt',map_location='cpu').get('step',0))" 2>/dev/null); st=${st:-0}
  [ "$st" -ge "$TARGET" ] && { echo "[wd] TARGET reached at step $st $(date)" >> $LOG; mirror; break; }
  echo "[wd] launch from step $st $(date)" >> $LOG
  CUDA_VISIBLE_DEVICES=$GPU python -u training/synthseg/train_synth.py \
    --labels /dev/shm/synthseg/label_maps --out $SHM --iters $TARGET \
    --batch 16 --patch 64 --sigma_max 0.05 --thick_max 8.0 --blur_max 1.0 \
    --save_every 10 --val_every 1000 --log_every 10 --grad_w 0.5 --res_cond 0 >> $LOG 2>&1 &
  PID=$!
  c=0
  while kill -0 $PID 2>/dev/null; do
    sleep 45; c=$((c+1))
    [ $((c % 6)) -eq 0 ] && mirror     # mirror to NFS ~every 4.5min (limit NFS-stall exposure)
    m=$(stat -c %Y $LOG 2>/dev/null || echo 0)
    [ $(( $(date +%s) - m )) -gt $STALL ] && { echo "[wd] STALL kill $PID $(date)" >> $LOG; kill -9 $PID 2>/dev/null; sleep 8; break; }
  done
  mirror
  sleep 30   # back off between relaunches so a repeatedly-failing launch can't storm the box
done
