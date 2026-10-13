import os, sys, time, cProfile, pstats, io, copy
sys.path.insert(0, "/home/amrl_sunny/ADDS_2/SAC_UED6_easy")
os.chdir("/home/amrl_sunny/ADDS_2/SAC_UED6_easy")
from configs import resolve_config
from learn.zero_shot import validation_levels, build_eval_model, validation_seed
from sim import robot_action as ra
cfg = resolve_config(check_data=False)
name, lv = validation_levels(cfg)[0]
lv = copy.deepcopy(lv); lv.robot_num = 3
m = build_eval_model(lv, validation_seed(name))
for rb in m.robots: ra.apply_to(rb, ra.encode((1.0, 0.5), "guide", speed=0.6))
for _ in range(30): m.step()
pr = cProfile.Profile(); pr.enable()
t = time.time()
for k in range(60):
    if k % 8 == 0:
        for rb in m.robots: ra.apply_to(rb, ra.encode((1.0, -0.5), "guide", speed=0.6))
    m.step()
dt = time.time() - t
pr.disable()
print("ms/step", dt / 60 * 1000, "agents", sum(1 for a in m.crowds if not a.dead))
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(35); print(s.getvalue()[:9000])
