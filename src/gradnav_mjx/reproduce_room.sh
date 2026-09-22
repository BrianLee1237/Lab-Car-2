#!/usr/bin/env bash
# Reproduce the 47% / 56% room policy in one command.
#
# Runs the original four legs back to back. The legs exist because the
# goal-distance curriculum is paced against --total-steps: a single
# 400k-step run trains on ~1.8-4.1m goals at 60k while the eval scores
# the full 5-10m band, so it reads ~5-8% and looks broken when it is
# only early. See TRAINING.md.
#
# Takes several hours. Safe to re-run: each leg resumes its checkpoint,
# so if you interrupt it, run the script again and it picks up.
set -euo pipefail
cd "$(dirname "$0")"

CKPT=${CKPT:-/tmp/room5.pkl}
CKPT4=${CKPT4:-/tmp/room_obs2.pkl}
CFG="--room --room-size 8.0 --n-inner 4 --n-humans 3 --spawn-half 6.5
     --min-dist 5.0 --max-dist 10.0 --horizon 300 --n-envs 16
     --warmup-steps 500 --seed 51"

# Match only real python invocations. A bare "train_sac.py" pattern
# also matches any shell, pgrep or tail whose command line merely
# mentions the file -- including a watcher in another terminal -- and
# would refuse to start for no reason. The [p] keeps this pgrep from
# matching itself.
if pgrep -f "[p]ython.*train_sac\.py" > /dev/null; then
  echo "ERROR: train_sac.py is already running. Legs share a checkpoint" >&2
  echo "and will corrupt each other. Stop it first:" >&2
  echo "  pkill -f 'python.*train_sac'" >&2
  exit 1
fi

leg () {  # leg <n> <total-steps> <eval-every> <eval-n> <ckpt> [extra...]
  local n=$1 tot=$2 every=$3 evn=$4 ck=$5; shift 5
  echo
  echo "===== leg $n -> $tot steps  (log: leg$n.log) ====="
  python -u train_sac.py $CFG --total-steps "$tot" --eval-every "$every" \
    --eval-n "$evn" --ckpt "$ck" --resume "$@" 2>&1 | tee "leg$n.log" \
    | grep -vE "Failed to import|^$"
}

leg 1  60000  5000 30 "$CKPT"
leg 2 160000 10000 30 "$CKPT"
leg 3 300000 20000 40 "$CKPT"

# Leg 4 raises the obstacle weight. The copy preserves the 300k state so
# that experiment cannot clobber it -- it is the same training lineage,
# and leg 4 should log "resumed from ... at 300000 steps".
cp "$CKPT" "$CKPT4"
leg 4 400000 20000 40 "$CKPT4" --obstacle-weight 2.0

# Deliberately NOT sac_policy_room_best.npz: that name holds the
# committed reference policy, and clobbering it would destroy the only
# copy of a known-good result with whatever this run happened to
# produce. Compare first, rename by hand if this run is better.
OUT=sac_policy_room_$(date +%Y%m%d_%H%M).npz
cp sac_policy_best.npz "$OUT"
echo
echo "===== training done. scoring $OUT at n=100 ====="
EVAL="--checkpoint $OUT --room --n-inner 4 --n-humans 3
      --spawn-half 6.5 --max-dist 10.0 --eval-n 100 --horizon 300 --bands 5,10"
echo; echo "--- training map (expect ~47%) ---"
python eval_band.py $EVAL --map-seed 51  2>&1 | grep -vE "Failed to import|^$"
echo; echo "--- held-out map (expect ~56%) ---"
python eval_band.py $EVAL --map-seed 200 2>&1 | grep -vE "Failed to import|^$"
echo
echo "Policy saved as $OUT. The committed reference policy"
echo "(sac_policy_room_best.npz, 47%/56%) was left untouched -- if this"
echo "run beat it, replace it by hand."
