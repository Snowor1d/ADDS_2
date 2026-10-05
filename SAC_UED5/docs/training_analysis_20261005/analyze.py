"""Read-only analysis of the October 4-5 run. No simulator or GPU work."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
LOG = Path('/home/amrl_sunny/Log_SAC_UED5_madrl')
CUTOFF = 2891
events = []
with (LOG / 'events.jsonl').open() as fh:
    for line in fh:
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue  # A live writer can leave its final line incomplete.
        if e.get('global_episode', 0) <= CUTOFF:
            events.append(e)
episodes = [e for e in events if e['kind'] == 'episode']
train = [e for e in events if e['kind'] == 'train']
groups = {'random_1_500': [e for e in episodes if e['global_episode'] <= 500],
          'recent_2501_2891': [e for e in episodes if e['global_episode'] >= 2501]}
summary = {'cutoff_episode': CUTOFF, 'episode_count': len(episodes)}
keys = ['total_reward', 'reward/person_time', 'reward/remaining_path',
        'reward/reentry', 'reward/collision', 'held_clear_success',
        'first_empty_step', 'robot_speed_mean', 'command_norm_mean']
for name, rows in groups.items():
    summary[name] = {'n': len(rows), **{
        k: float(np.mean([e['metrics']['episode/' + k] for e in rows]))
        for k in keys}}

# Adjust the observational comparison for site, hazard area and initial crowd.
# This cannot replace evaluation with identical levels and random seeds.
rows = groups['random_1_500'] + groups['recent_2501_2891']
X = np.array([[1, int(e['global_episode'] >= 2501),
               e['metrics']['episode/hazard_area_fraction'],
               e['metrics']['episode/initial_inside'],
               e['metrics']['episode/initial_population'],
               *[int(e['metrics']['episode/site_index'] == i)
                 for i in range(1, 20)]] for e in rows], dtype=float)
inverse = np.linalg.pinv(X.T @ X)
adjusted = {}
for key in ['total_reward', 'reward/person_time', 'reward/collision']:
    y = np.array([e['metrics']['episode/' + key] for e in rows])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    residual = y - X @ beta
    leverage = np.einsum('ij,jk,ik->i', X, inverse, X)
    weights = (residual / np.maximum(1 - leverage, 1e-6)) ** 2
    cov = inverse @ (X.T @ (X * weights[:, None])) @ inverse
    se = float(np.sqrt(cov[1, 1]))
    adjusted[key] = {'recent_minus_random': float(beta[1]),
                     'approx_HC3_95_interval': [float(beta[1] - 1.96*se),
                                               float(beta[1] + 1.96*se)]}
summary['observational_adjustment'] = adjusted

with np.load(LOG / 'replay_buffer.npz', allow_pickle=False) as data:
    action, valid = data['action'], data['has_action']
    pose, next_index = data['pose'], data['next']
    summary['replay'] = {'records': len(action), 'blocks': {}}
    for name, start, end in [('first_125000', 0, 125000),
                             ('last_125500', len(action)-125500, len(action))]:
        ix = np.arange(start, end)
        ix = ix[valid[ix]]
        move = action[ix, :, :2]
        i = ix[(next_index[ix] >= 0) & (next_index[ix] < len(action))]
        distance = np.linalg.norm(pose[next_index[i]] - pose[i], axis=-1)
        summary['replay']['blocks'][name] = {
            'actions': len(ix), 'axis_abs_gt_1_9': float(np.mean(abs(move)>1.9)),
            'command_norm_gt_1': float(np.mean(np.linalg.norm(move, axis=-1)>1)),
            'displacement_lt_0_1_m': float(np.mean(distance<0.1)),
            'mean_displacement_m': float(distance.mean())}

def series(rows, key, axis):
    return ([e[axis] for e in rows],
            np.array([e['metrics'][key] for e in rows]))

fig, axs = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
for ax, key, title in zip(axs.flat[:4],
        ['total_reward', 'reward/collision', 'robot_speed_mean', 'held_clear_success'],
        ['Return (higher is better)', 'Collision reward (higher is better)',
         'Actual robot speed (m/s)', '60-step clear success rate']):
    x, y = series(episodes, 'episode/'+key, 'global_episode')
    ax.plot(x, y, alpha=.15, lw=.5)
    ax.plot(x[99:], np.convolve(y, np.ones(100)/100, mode='valid'), lw=2)
    ax.axvline(500, color='black', linestyle='--', alpha=.5)
    ax.set(title=title, xlabel='Episode')
x, y = series(train, 'train/alpha', 'global_update')
axs[2, 0].semilogy(x, y)
axs[2, 0].set(title='Entropy temperature alpha', xlabel='Gradient update')
x, y = series(train, 'train/loss_q', 'global_update')
axs[2, 1].plot(x, y)
axs[2, 1].set(title='Twin critic MSE sum', xlabel='Gradient update')
fig.suptitle('SAC_UED5 dataset run: through episode 2891')
fig.savefig(ROOT / 'learning_diagnostics.png', dpi=150)
(ROOT / 'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
