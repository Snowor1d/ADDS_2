"""Parameter counts, update time and training VRAM of every network option.

    python3 -m cli.measure_networks                    # all families and sizes, batch 128
    python3 -m cli.measure_networks --batch 256 --sharing critic

Each row builds the full learner (actor, CRITIC_ENSEMBLE critics, their
targets, AdamW states) on the GPU and runs SAC updates on random batches;
the VRAM figure is torch.cuda.max_memory_allocated during those updates plus
the CUDA context (measured separately), i.e. what `nvidia-smi` would show
for the learner process. Worker inference is one CPU-thread actor call.
"""

import argparse
import sys
import time


def main(argv=None):
    sys.stdout.reconfigure(line_buffering=True)  # progress shows up under nohup
    from configs import apply_cli_overrides
    argv = apply_cli_overrides(list(sys.argv[1:] if argv is None else argv))
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--sharing", default="separate")
    ap.add_argument("--updates", type=int, default=30)
    ap.add_argument("--families", nargs="*", default=["legacy_rv3", "cnn", "impala"])
    ap.add_argument("--sizes", nargs="*", default=["s", "m", "l", "xl"])
    args = ap.parse_args(argv)

    import torch
    from configs import resolve_config
    from learn.networks import build_policy, parameter_count
    from learn.sac import SACLearner

    dev = "cuda"
    torch.cuda.init()
    torch.zeros(1, device=dev)
    import subprocess
    ctx_mb = None
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True)
        import os
        for line in out.stdout.strip().splitlines():
            pid, mem = [x.strip() for x in line.split(",")]
            if int(pid) == os.getpid():
                ctx_mb = float(mem)
    except Exception:
        pass
    rows = []
    for fam in args.families:
        sizes = ["-"] if fam == "legacy_rv3" else args.sizes
        for size in sizes:
            over = {"NET_ENCODER": fam, "BATCH_SIZE": args.batch}
            if fam != "legacy_rv3":
                over.update(NET_SIZE=size, ENCODER_SHARING=args.sharing)
            cfg = resolve_config(overrides=over)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            L = SACLearner(cfg, dev)
            B = args.batch
            C = cfg.obs_channels()
            E, G = cfg.EGO_MAP_SIZE, cfg.DOWNSAMPLE_MAP_SIZE

            def batch():
                return {"ego": torch.rand(B, C, E, E, device=dev), "glob": torch.rand(B, C, G, G, device=dev),
                        "state": torch.rand(B, 5, device=dev), "action": torch.rand(B, 2, device=dev) * 4 - 2,
                        "reward": -torch.rand(B, device=dev), "discount": torch.full((B,), 0.99, device=dev),
                        "done": torch.zeros(B, device=dev),
                        "ego2": torch.rand(B, C, E, E, device=dev), "glob2": torch.rand(B, C, G, G, device=dev),
                        "state2": torch.rand(B, 5, device=dev)}
            for _ in range(3):
                L.update(batch())
            torch.cuda.synchronize()
            t = time.time()
            for _ in range(args.updates):
                L.update(batch())
            torch.cuda.synchronize()
            upd = (time.time() - t) / args.updates
            peak = (torch.cuda.max_memory_allocated() - base) / 2 ** 20
            reserved = torch.cuda.max_memory_reserved() / 2 ** 20
            pa = parameter_count(L.actor) if not L.critic.shared else parameter_count(L.actor.head) + parameter_count(L.critic.encoder)
            pc = parameter_count(L.critic) / cfg.CRITIC_ENSEMBLE
            pol = build_policy(cfg).eval()
            torch.set_num_threads(1)
            e = torch.rand(1, C, E, E)
            g = torch.rand(1, C, G, G)
            s = torch.rand(1, 3)
            with torch.no_grad():
                for _ in range(5):
                    pol.sample(e, g, s)
                t = time.time()
                for _ in range(50):
                    pol.sample(e, g, s)
            inf = (time.time() - t) / 50
            rows.append((fam, size, pa / 1e6, pc / 1e6, upd, peak, reserved, inf * 1000))
            print(f"{fam:10s} {size:2s} actor {pa/1e6:6.2f} M  critic {pc/1e6:6.2f} M  "
                  f"update {upd*1000:6.1f} ms  peak alloc {peak:7.0f} MB  reserved {reserved:7.0f} MB  "
                  f"worker {inf*1000:5.1f} ms", flush=True)
            del L, pol
    print()
    print(f"CUDA context ~{ctx_mb} MB (add to the reserved figure for nvidia-smi)")
    print("| 계열 | 크기 | 액터 | 크리틱 1개 | 업데이트 1회 | 학습 VRAM(할당 피크) | 예약 피크 | 워커 추론 |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for fam, size, pa, pc, upd, peak, res, inf in rows:
        print(f"| {fam} | {size} | {pa:.2f} M | {pc:.2f} M | {upd*1000:.0f} ms | {peak/1024:.2f} GB | "
              f"{res/1024:.2f} GB | {inf:.1f} ms |")


if __name__ == "__main__":
    main()
