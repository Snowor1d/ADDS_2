#!/bin/bash
# Probe states off the training crops, for the spatial-critic check
# (critic_diagnosis.md section 9): three unseen sites at 200 m (branch at 40 s
# and 160 s) and the training sites at 100 m (branch at 40 s). 10 repeats each,
# under the checkpoint's own gamma and decision length.
export CK=${CK:?checkpoint}
export OVR='{"GAMMA_WAYPOINT": 0.995, "ROBOT_DECISION_MAX_S_WAYPOINT": 20}'
cd "$(dirname "$0")"
export R=$PWD/results/gen
mkdir -p "$R"
{ for t0 in 80 320; do for s in kreuzberg:200:0 gracia:200:0 trastevere:200:0; do echo "$s $t0"; done; done
  for s in hongdae:100:0 hongdae:100:1 hongdae:100:2 dotonbori:100:0 dotonbori:100:1 dotonbori:100:2; do echo "$s 80"; done; } |
  xargs -P "${JOBS:-7}" -L 1 bash -c 'n=$(echo "$0" | tr ":" "_"); out="$R/s_${n}_t$1.json"; [ -f "$out" ] || ${DRY:+echo} nice -n 10 python3 true_adv_v2.py "$0" "$1" "$CK" "$out" --reps 10 --overrides "$OVR" > "$R/s_${n}_t$1.log" 2>&1'
echo BATCH_DONE
