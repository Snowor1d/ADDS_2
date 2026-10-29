#!/bin/bash
# Join the chunks of run_chunks.sh into results/gen/s_<state>_t80.json.
cd "$(dirname "$0")"
G=$PWD/results/gen
for s in kreuzberg_200_0 gracia_200_0 trastevere_200_0; do
  b=s_${s}_t80
  for k in 0 1 2 3 4; do [ -f "$G/chunks/${b}_c$k.json" ] || { echo "missing ${b}_c$k"; exit 1; }; done
  cp "$G/chunks/${b}_c0.json" "$G/$b.json"
  for k in 1 2 3 4; do cp "$G/chunks/${b}_c$k.json" "$G/${b}_more.json"; python3 merge_reps.py "$G"; done
done
