"""Re-score saved testbed critics on the current probe states (for example
after merge_reps.py added repeats), without retraining.

Usage: eval_saved.py SAVE_DIR CKPT OUT.jsonl [RESULTS_DIR]
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import testbed as T  # noqa: E402

import torch

from configs import resolve_config

save_dir, ckpt, out = sys.argv[1], sys.argv[2], sys.argv[3]
rdir = sys.argv[4] if len(sys.argv) > 4 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = resolve_config(check_data=False)
probes = T.load_probes(rdir)
ck = torch.load(ckpt, map_location="cpu", weights_only=False)
rows = []
ref1 = T.make_critic(cfg, "mse", None, ck["q1"], dev); ref1.load_state_dict(ck["q1"])
ref2 = T.make_critic(cfg, "mse", None, ck["q2"], dev); ref2.load_state_dict(ck["q2"])
ref1.eval(); ref2.eval()
rows.append({"variant": "checkpoint", "update": -1, **T.probe_metrics(ref1, ref2, probes, dev)})
for path in sorted(glob.glob(os.path.join(save_dir, "*_u*.pt"))):
    s = torch.load(path, map_location="cpu", weights_only=False)
    q = []
    for k in ("q1", "q2"):
        c = T.VariantCritic(cfg, T.ra.ACTION_DIM, s["kind"], s["support"]).to(dev)
        c.load_state_dict(s[k]); c.eval(); q.append(c)
    rows.append({"variant": f"{s['kind']}:{s['n']}", "update": s["update"],
                 **T.probe_metrics(q[0], q[1], probes, dev)})
with open(out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "probe_rows"})
