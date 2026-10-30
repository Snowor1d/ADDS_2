"""Re-score critics saved by testbed_spatial.py on every probe state there is
now (for example after the generalization probes finished), without
retraining. Writes rows in testbed_spatial.py's format, so
analyze_spatial.py reads either.

Usage: eval_spatial_saved.py SAVE_DIR CKPT OUT.jsonl --overrides JSON
"""
import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import testbed_spatial as T  # noqa: E402  (chdirs to the project root)

import torch

from configs import resolve_config
from learn.networks import CentralizedCritic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("save_dir")
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--overrides", default="")
    ap.add_argument("--probe-dirs", default=",".join([os.path.join(T.HERE, "results"),
                                                      os.path.join(T.HERE, "results", "gen")]))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.3)
    cfg = resolve_config(json.loads(args.overrides) if args.overrides else None, check_data=False)
    probes = T.load_probes(args.probe_dirs.split(","))
    print(f"{len(probes)} probe states", flush=True)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    with open(args.out, "w") as f:
        ref = []
        for k in ("q1", "q2"):
            c = CentralizedCritic(cfg, T.ra.ACTION_DIM).to(dev)
            c.load_state_dict(ck[k])
            ref.append(c.eval())
        f.write(json.dumps({"update": -1, "variant": "checkpoint",
                            "probe": T.probe_values(*ref, probes, dev)}) + "\n")
        paths = sorted(glob.glob(os.path.join(args.save_dir, "*_u*.pt")),
                       key=lambda p: (re.sub(r"_u\d+\.pt$", "", p), int(re.search(r"_u(\d+)\.pt$", p).group(1))))
        for path in paths:
            s = torch.load(path, map_location="cpu", weights_only=False)
            q = []
            for k in ("q1", "q2"):
                c = (T.SpatialCritic(cfg, T.ra.ACTION_DIM) if s["kind"] == "spatial"
                     else CentralizedCritic(cfg, T.ra.ACTION_DIM)).to(dev)
                c.load_state_dict(s[k])
                q.append(c.eval())
            name = s.get("name", s["kind"])
            f.write(json.dumps({"update": s["update"], "variant": name,
                                "probe": T.probe_values(*q, probes, dev)}) + "\n")
            f.flush()
            print(path, flush=True)


if __name__ == "__main__":
    main()
