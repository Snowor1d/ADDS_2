"""Replay suite scenarios exactly as cli.evaluate ran them and record each one
as a video and a per-step trace for analysis.

    python3 -m cli.trace_episode --checkpoint ~/RV4_checkpoints/aug_cnn_l_ep3800.pth \
        --ids test/map1504/s4 test/map1504/s2 --mode stoch --out ~/RV4_eval/traces/augL-ep3800
    python3 -m cli.trace_episode --checkpoint X --failures-from ~/RV4_eval/augL-ep3800-1029-1000 \
        --mode stoch --out ~/RV4_eval/traces/augL-ep3800 [--maps map1504 map1503] [--no-video]

Seeding follows evaluation/runner.run_job, so a replay reproduces the
evaluated episode. With --failures-from the replay is checked against the
recorded result (evac time, people left, collisions) and the check is
written to the summary.

Output in --out, per scenario (id with "/" -> "__"):
    <id>.mp4   every --every-th step, robot in red, robot followers in orange
    <id>.npz   per simulation step (0.5 s):
                 robot_xy (T,2), robot_vel (T,2) from position change (m/s),
                 collision (T,), alive (T,),
                 followers (T,)  alive people currently following the robot (type 0)
                 my_way (T,), neighbor_followers (T,)  types 1 and 2
               per robot decision (every ACTION_SCALE steps):
                 dec_step (K,), action (K,2) applied, mean_action (K,2) policy
                 mean squashed, std_u (K,2) pre-squash std, state (K,d) robot state,
                 crowd_xy (K,N,2), crowd_type (K,N), crowd_out (K,N) evacuated
    summary.jsonl  one row per scenario: result, replay check, map geometry
                   (obstacle and exit polygons, size)
"""

import argparse
import glob
import json
import os
import random
import sys

import numpy as np


def _job(args):
    sc, checkpoint, det, out_dir, video, every, expected = args
    import torch
    torch.set_num_threads(1)
    from configs import cfg
    from evaluation.suites import scenario_spec
    from learn.policy_io import load_policy
    from learn.obs import network_state
    from learn.rollout import (PolicyController, build_env, eval_step_cap,
                               freeflow_steps, run_episode)

    actor, pcfg, _ = load_policy(checkpoint)
    rec = {k: [] for k in ("robot_xy", "robot_vel", "collision", "alive", "followers", "my_way",
                           "neighbor_followers", "dec_step", "action", "mean_action", "std_u",
                           "state", "crowd_xy", "crowd_type", "crowd_out")}
    step_now = [0]
    last_xy = [None]

    class Tracing(PolicyController):
        @torch.no_grad()
        def act(self, env, ob, stack, state5):
            ego, glob_ = stack.network_input(ob.glob_static)
            ego_t = torch.from_numpy(ego)[None]
            glob_t = torch.from_numpy(glob_)[None]
            st = torch.from_numpy(network_state(state5, self.use_scale))[None]
            mean, log_std = self.actor(ego_t, glob_t, st)   # no RNG use
            a = super().act(env, ob, stack, state5)
            rec["dec_step"].append(step_now[0])
            rec["action"].append(a.copy())
            rec["mean_action"].append((4.0 * torch.sigmoid(mean[0]) - 2.0).numpy())
            rec["std_u"].append(log_std[0].exp().numpy())
            rec["state"].append(np.asarray(state5, dtype=np.float32).ravel())
            rec["crowd_xy"].append(np.array([c.xy[:2] for c in env.crowds], dtype=np.float32))
            rec["crowd_type"].append(np.array([int(c.type) for c in env.crowds], dtype=np.int8))
            rec["crowd_out"].append(np.array([bool(c.dead) for c in env.crowds]))
            return a

    # same order and seeds as evaluation/runner.run_job
    ctrl = Tracing(actor, pcfg, deterministic=det, seed=sc["seed"])
    env = build_env(scenario_spec(sc), cfg, robot="Q")
    ff = freeflow_steps(env, cfg)
    dt = float(cfg.AGENT_TIME_STEP)
    last_xy[0] = list(env.robot.xy[:2])
    random.seed(int(sc["seed"]) + 1)
    np.random.seed((int(sc["seed"]) + 1) % (2 ** 32))

    name = sc["id"].replace("/", "__")
    writer = rend = None
    if video:
        from continuous_renderer import ContinuousRenderer
        from evaluation.video import _FFmpeg
        rend = ContinuousRenderer(world_size=(float(env.width), float(env.height)),
                                  crowd_colors={0: "#ffa500", 1: "#4e79a7", 2: "#4e79a7"},
                                  robot_color="#e15759", show_agent_heading=False,
                                  show_robot_heading=False, trail_target="robot", trail_style="fade",
                                  max_trail=400, single_color_edges=True, exit_size=5.0,
                                  snap_exit_to_boundary=False)
        writer = _FFmpeg(os.path.join(out_dir, name + ".mp4"), 20)

    def on_step(e, step):
        step_now[0] = step
        r = e.robot
        rec["robot_xy"].append(r.xy[:2])
        # r.vel is not maintained by the simulator for the robot: use the
        # position change over the step instead
        prev = last_xy[0] if last_xy[0] is not None else r.xy[:2]
        rec["robot_vel"].append([(r.xy[0] - prev[0]) / dt, (r.xy[1] - prev[1]) / dt])
        last_xy[0] = list(r.xy[:2])
        rec["collision"].append(int(bool(getattr(r, "collision_check", 0))))
        live = [c for c in e.crowds if not c.dead]
        rec["alive"].append(len(live))
        rec["followers"].append(sum(c.type == 0 for c in live))
        rec["my_way"].append(sum(c.type == 1 for c in live))
        rec["neighbor_followers"].append(sum(c.type == 2 for c in live))
        if writer is not None and step % every == 0:
            writer.write(rend.draw(e, step=step))

    try:
        res = run_episode(env, ctrl, pcfg, max_steps=eval_step_cap(cfg, env), on_step=on_step)
    finally:
        if writer is not None:
            writer.close()
    arrays = {}
    for k, v in rec.items():
        dt = np.int16 if k in ("collision", "alive", "followers", "my_way", "neighbor_followers",
                                "dec_step") else None
        arrays[k] = np.asarray(v, dtype=dt) if dt else np.asarray(v, dtype=np.float32) \
            if k not in ("crowd_type", "crowd_out") else np.asarray(v)
    np.savez_compressed(os.path.join(out_dir, name + ".npz"), **arrays)
    row = {"id": sc["id"], "map_key": sc["map_key"], "seed": sc["seed"],
           "augmentation": sc.get("augmentation"), "mode": "deterministic" if det else "stochastic",
           "checkpoint": checkpoint, "freeflow_steps": ff, **res,
           "width": float(env.width), "height": float(env.height),
           "exits": [[list(map(float, p)) for p in poly] for poly in env.exit_list],
           "obstacles": [[list(map(float, p)) for p in poly] for poly in env.obstacles]}
    if expected is not None:
        keys = ("evac_time_100", "alive_end", "robot_collision_steps")
        row["replay_matches_eval"] = all(abs(float(res[k]) - float(expected[k])) < 1e-6 for k in keys)
        row["eval_result"] = {k: expected[k] for k in keys}
    return row


def main(argv=None):
    sys.stdout.reconfigure(line_buffering=True)
    from configs import apply_cli_overrides
    argv = apply_cli_overrides(list(sys.argv[1:] if argv is None else argv))
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--suite", default="test")
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--failures-from", help="cli.evaluate output dir: trace its failed scenarios")
    ap.add_argument("--maps", nargs="*", help="with --failures-from: only these map keys")
    ap.add_argument("--mode", choices=("stoch", "det"), default="stoch")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--every", type=int, default=4, help="video frame every N steps")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    from configs import cfg
    from evaluation.suites import load_suite

    suite = load_suite(args.suite, cfg)
    by_id = {s["id"]: s for s in suite["scenarios"]}
    cond = "policy_stoch" if args.mode == "stoch" else "policy_det"
    expected = {}
    if args.failures_from:
        path = glob.glob(os.path.join(os.path.expanduser(args.failures_from), f"{args.suite}_results.jsonl"))[0]
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("condition") == cond and "error" not in r:
                    expected[r["id"]] = r
    ids = list(args.ids or [])
    if args.failures_from and not ids:
        ids = sorted(i for i, r in expected.items() if not r["success"]
                     and (not args.maps or r["map_key"] in args.maps))
    if not ids:
        raise SystemExit("no scenarios: give --ids or --failures-from")
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise SystemExit(f"not in suite {args.suite}: {missing}")
    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)
    ckpt = os.path.abspath(os.path.expanduser(args.checkpoint))
    jobs = [(by_id[i], ckpt, args.mode == "det", out, not args.no_video, args.every, expected.get(i))
            for i in ids]
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(max(1, min(args.workers, len(jobs)))) as pool, \
            open(os.path.join(out, "summary.jsonl"), "a", encoding="utf-8") as fh:
        for row in pool.imap_unordered(_job, jobs):
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(f"{row['id']}: success={row['success']:.0f} t100={row['evac_time_100']:.0f} "
                  f"left={row['alive_end']:.0f} replay_ok={row.get('replay_matches_eval')}")


if __name__ == "__main__":
    main()
