#!/bin/bash
# Start the testbed once every probe state is rebuilt and the shared GPU has room.
cd "$(dirname "$0")"
R=$PWD/results
until [ $(ls $R/s*_t*_state.npz 2>/dev/null | wc -l) -ge 18 ]; do sleep 60; done
until [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) -ge 4500 ]; do sleep 120; done
echo "launch $(date)"
setsid nohup python3 testbed_multi.py --variants mse:1,hlgauss:1,dueling:1,hlgauss_dueling:1,mse:3 \
  --ckpt /tmp/claude-1000/-home-amrl-sunny-ADDS-2/51a9f103-9bd2-4a2a-80bb-843dc94d52f2/scratchpad/ckpt_ul6kds66_ep3800.pth \
  --replay $HOME/Log_SAC_UED6_easy_fullinfo_clock_pbrs/replay_buffer.npz \
  --static-dir $HOME/Log_SAC_UED6_easy_fullinfo_clock_pbrs/replay_static \
  --probes $R --out $R/testbed.jsonl --updates 20000 --eval-every 1000 \
  --save-every 5000 --save-dir /tmp/claude-1000/-home-amrl-sunny-ADDS-2/51a9f103-9bd2-4a2a-80bb-843dc94d52f2/scratchpad/testbed_critics > $R/testbed.log 2>&1 < /dev/null &
