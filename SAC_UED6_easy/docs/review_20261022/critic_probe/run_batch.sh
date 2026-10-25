#!/bin/bash
# 18 probe states: 9 validation scenarios x branch at 40 s and 160 s.
CK=${CK:-$HOME/Log_SAC_UED6_easy_fullinfo_clock_pbrs/sac_checkpoint_ep_3800.pth}
cd "$(dirname "$0")"
for t0 in 80 320; do for s in 0 1 2 3 4 5 6 7 8; do echo "$s $t0"; done; done |
  xargs -P "${JOBS:-5}" -L 1 bash -c 'out=$PWD/results/s$0_t$1.json; [ -f $out ] || nice -n 10 python3 true_adv_v2.py $0 $1 '"$CK"' $out > $PWD/results/s$0_t$1.log 2>&1'
echo BATCH_DONE
