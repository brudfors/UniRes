#!/bin/bash
# Watchdog burst-loop for the FULL-IMAGE synth-restore model (whole-brain, batch 1).
# Same resilience as burst_synth.sh (graceful kill + wait_gpu_free + model-only resume),
# but runs `--full_image` instead of patch training. RESUME_OPT=0 = model-only resume
# (skip the 272MB opt-state load that wedges resume on this pod).
GPU=${GPU:-0}
RUN=${RUN:-synth_full}
SHM=/dev/shm/synthseg/runs/$RUN
NFS=/workspace/data/synthseg_restore/runs/$RUN
LOG=$SHM/train.log
TARGET=${TARGET:-40000}
STALL=${STALL:-150}
mkdir -p $SHM $NFS /dev/shm/synthseg/label_maps
cp -n /workspace/data/synthseg_restore/label_maps/*.nii.gz /dev/shm/synthseg/label_maps/ 2>/dev/null
if [ ! -f $SHM/last.pt ] && [ -f $NFS/last.pt ]; then cp $NFS/last.pt $SHM/last.pt 2>/dev/null; fi
cd /workspace/code/UniRes
export PYTHONPATH=/workspace/code/UniRes
export RESUME_OPT=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mirror() { [ -f $SHM/last.pt ] && cp -f $SHM/last.pt $NFS/last.pt 2>/dev/null; cp -f $LOG $NFS/train.log 2>/dev/null; }
graceful_kill() { local pid=$1; kill -TERM $pid 2>/dev/null; for i in $(seq 1 15); do kill -0 $pid 2>/dev/null || return; sleep 1; done; echo "[wd] SIGKILL $pid" >> $LOG; kill -9 $pid 2>/dev/null; sleep 2; }
wait_gpu_free() { for i in $(seq 1 30); do u=$(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' '); [ -z "$u" ] && return; [ "$u" -lt 1500 ] && { echo "[wd] GPU$GPU free (${u}MiB)" >> $LOG; return; }; sleep 2; done; }
while true; do
  st=$(python -c "import torch;print(torch.load('$SHM/last.pt',map_location='cpu').get('step',0))" 2>/dev/null); st=${st:-0}
  [ "$st" -ge "$TARGET" ] && { echo "[wd] TARGET reached at $st $(date)" >> $LOG; mirror; break; }
  # ATTACH-GUARD: if a training proc for THIS run (--out $SHM, exact, trailing space avoids
  # synth_full matching synth_full_cond) is already alive, monitor it instead of double-launching.
  EXIST=$(pgrep -f -- "--out $SHM " | head -1)
  if [ -n "$EXIST" ]; then
    echo "[wd] attach existing PID $EXIST at step $st $(date)" >> $LOG; PID=$EXIST
  else
  wait_gpu_free
  echo "[wd] launch from step $st $(date)" >> $LOG
  CUDA_VISIBLE_DEVICES=$GPU python -u training/synthseg/train_synth.py \
    --full_image 1 --full_size ${FULL_SIZE:-192} --labels /dev/shm/synthseg/label_maps \
    --out $SHM --iters $TARGET --sigma_max 0.05 --thick_max ${THICK_MAX:-8.0} --inplane_max ${INPLANE_MAX:-0.0} --blur_max 1.0 \
    --save_every ${SAVE_EVERY:-100} --val_every ${VAL_EVERY:-2000} --log_every 20 \
    --fixed_lr ${FIXED_LR:-0} --lr ${LR:-1e-4} --net ${NET:-biasfree3d} --grad_w 0.5 --res_cond ${RES_COND:-0} \
    \
    --realism ${REALISM:-0.0} --zero_bg ${ZERO_BG:-0} >> $LOG 2>&1 &
  PID=$!
  fi
  c=0
  while kill -0 $PID 2>/dev/null; do
    sleep 30; c=$((c+1))
    [ $((c % 9)) -eq 0 ] && mirror
    m=$(stat -c %Y $LOG 2>/dev/null || echo 0)
    [ $(( $(date +%s) - m )) -gt $STALL ] && { echo "[wd] STALL kill $PID $(date)" >> $LOG; graceful_kill $PID; break; }
  done
  mirror
  sleep 20
done
