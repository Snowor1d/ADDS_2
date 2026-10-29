#!/bin/bash
# One runner over the slow generalization probes (200 m, branch at 40 s),
# split into 2-repeat chunks so several runners share them. A chunk is
# claimed with mkdir, so runners can be added at any time; merge_chunks.sh
# joins the chunks when all are done.
export CK=${CK:?checkpoint}
OVR='{"GAMMA_WAYPOINT": 0.995, "ROBOT_DECISION_MAX_S_WAYPOINT": 20}'
cd "$(dirname "$0")"
C=$PWD/results/gen/chunks
mkdir -p "$C/locks"
for k in 0 1 2 3 4; do for s in kreuzberg:200:0 gracia:200:0 trastevere:200:0; do
  n=$(echo "$s" | tr ":" "_")
  name=s_${n}_t80_c$k
  mkdir "$C/locks/$name" 2>/dev/null || continue
  nice -n 19 python3 true_adv_v2.py "$s" 80 "$CK" "$C/$name.json" --reps 2 --rep-start $((2 * k)) \
    --overrides "$OVR" > "$C/$name.log" 2>&1
done; done
