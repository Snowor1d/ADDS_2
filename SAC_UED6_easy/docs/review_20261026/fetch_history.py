"""Read-only W&B scalar history export, compressed and without null padding."""
import gzip, json, math, sys
from pathlib import Path
import wandb
root = Path(__file__).resolve().parent
api = wandb.Api(timeout=120, overrides={"base_url": "https://api.wandb.ai"})
for rid in sys.argv[1:]:
    r = api.run("437snowcap-sungkyunkwan-university/adds-sac-ued6-261011ENV/" + rid)
    path = root / (rid + ".jsonl.gz"); temp = path.with_suffix(path.suffix + ".part")
    n = 0
    with gzip.open(temp, "wt") as f:
        for row in r.scan_history(page_size=1500):
            clean = {k: v for k, v in row.items() if isinstance(v, (int, float, str, bool))
                     and not (isinstance(v, float) and not math.isfinite(v))}
            f.write(json.dumps(clean) + "\n"); n += 1
    temp.replace(path)
    print(rid, r.name, n, flush=True)
