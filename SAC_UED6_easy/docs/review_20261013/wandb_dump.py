import wandb, json, sys, os, math
api = wandb.Api(timeout=120)
ent = api.default_entity
out = sys.argv[1]
targets = [("adds-sac-ued6-261011ENV", None), ("adds-sac-ued6", None)]
for proj, _ in targets:
    for r in api.runs(f"{ent}/{proj}", order="-created_at"):
        fn = os.path.join(out, f"{proj}__{r.id}.jsonl")
        cfn = os.path.join(out, f"{proj}__{r.id}.config.json")
        with open(cfn, "w") as f:
            json.dump({"name": r.name, "state": r.state, "created_at": r.created_at,
                       "config": {k: v for k, v in r.config.items() if "key" not in k.lower() and "token" not in k.lower()},
                       "summary": {k: v for k, v in r.summary._json_dict.items() if isinstance(v, (int, float, str, bool)) or v is None}},
                      f, default=str)
        n = 0
        with open(fn, "w") as f:
            for row in r.scan_history(page_size=1000):
                clean = {}
                for k, v in row.items():
                    if isinstance(v, (int, float, str, bool)) or v is None:
                        if isinstance(v, float) and not math.isfinite(v):
                            continue
                        clean[k] = v
                f.write(json.dumps(clean) + "\n")
                n += 1
        print(proj, r.id, r.name, n, flush=True)
