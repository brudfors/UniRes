#!/bin/bash
# Watchdog burst-loop for SINGLE-GPU synth-restore training.
# Trains to /dev/shm (tmpfs; NFS /workspace stalls under load -> livelock), and MIRRORS
# last.pt/best.pt to persistent NFS every ~45s so a pod reset never loses hours.
# DDP is avoided on purpose: on this CPU-saturated shared box NCCL allreduce starves and
# collapses throughput to <0.15 it/s, whereas a single process runs at ~1.1 it/s.
GPU=${GPU:-0}
RUN=${RUN:-synth_thick}
SHM=/dev/shm/synthseg/runs/$RUN
NFS=/workspace/data/synthseg_restore/runs/$RUN
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

# GRACEFUL kill: SIGTERM first so CUDA can release the context (avoids ORPHANED GPU memory,
# the real cause of the "resume hangs at 43GB" wedge), wait up to 15s, SIGKILL only if stuck.
graceful_kill() {
  local pid=$1
  kill -TERM $pid 2>/dev/null
  for i in $(seq 1 15); do kill -0 $pid 2>/dev/null || return; sleep 1; done
  echo "[wd] SIGTERM timed out -> SIGKILL $pid $(date)" >> $LOG
  kill -9 $pid 2>/dev/null; sleep 2
}
# Wait until the GPU's memory is actually FREED before relaunching (a fresh process that
# resumes onto a still-occupied GPU is what wedges at 43GB). Give up after ~60s regardless.
wait_gpu_free() {
  for i in $(seq 1 30); do
    u=$(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    [ -z "$u" ] && return
    [ "$u" -lt 1500 ] && { echo "[wd] GPU$GPU free (${u}MiB) $(date)" >> $LOG; return; }
    sleep 2
  done
  echo "[wd] WARN GPU$GPU still ${u}MiB after 60s $(date)" >> $LOG
}
while true; do
  st=$(python -c "import torch;print(torch.load('$SHM/last.pt',map_location='cpu').get('step',0))" 2>/dev/null); st=${st:-0}
  [ "$st" -ge "$TARGET" ] && { echo "[wd] TARGET reached at step $st $(date)" >> $LOG; mirror; break; }
  # ATTACH-GUARD: monitor an already-running proc for THIS run instead of double-launching.
  EXIST=$(pgrep -f -- "--out $SHM " | head -1)
  if [ -n "$EXIST" ]; then
    echo "[wd] attach existing PID $EXIST at step $st $(date)" >> $LOG; PID=$EXIST
  else
  wait_gpu_free                       # ensure the GPU is actually free before (re)launching
  echo "[wd] launch from step $st $(date)" >> $LOG
  # BATCH/SAVE_EVERY/LOG_EVERY are env-overridable so a heavily-contended box can RATCHET:
  #  small batch (fewer per-step CUDA launches) + save/log every 2 steps means the process
  #  checkpoints + shows liveness every couple steps, so it accumulates forward across
  #  load windows instead of losing a whole 10-step interval on each STALL-kill.
  CUDA_VISIBLE_DEVICES=$GPU python -u training/synthseg/train_synth.py \
    --labels /dev/shm/synthseg/label_maps --out $SHM --iters $TARGET \
    --batch ${BATCH:-8} --patch 64 --sigma_max 0.05 --thick_max ${THICK_MAX:-8.0} --inplane_max ${INPLANE_MAX:-0.0} --blur_max 1.0 \
    --save_every ${SAVE_EVERY:-2} --val_every ${VAL_EVERY:-1000} --log_every ${LOG_EVERY:-2} \
    --fixed_lr ${FIXED_LR:-0} --lr ${LR:-2e-4} --net ${NET:-biasfree3d} \
    --grad_w 0.5 --res_cond ${RES_COND:-0} \
    --op_consistent ${OP_CONSISTENT:-0} --recon_iters ${RECON_ITERS:-3} --texture ${TEXTURE:-0.0} --realism ${REALISM:-0.0} --zero_bg ${ZERO_BG:-0} --multi_ori ${MULTI_ORI:-0.0} --real_frac ${REAL_FRAC:-0.0} >> $LOG 2>&1 &
  PID=$!
  fi
  c=0
  while kill -0 $PID 2>/dev/null; do
    sleep 45; c=$((c+1))
    [ $((c % 6)) -eq 0 ] && mirror     # mirror to NFS ~every 4.5min (limit NFS-stall exposure)
    m=$(stat -c %Y $LOG 2>/dev/null || echo 0)
    [ $(( $(date +%s) - m )) -gt $STALL ] && { echo "[wd] STALL kill $PID $(date)" >> $LOG; graceful_kill $PID; break; }
  done
  mirror
  sleep 30   # back off between relaunches so a repeatedly-failing launch can't storm the box
done
