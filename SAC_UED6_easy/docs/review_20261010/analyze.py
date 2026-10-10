"""Analyze the downloaded scalar history; no training or external writes."""
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/adds-review-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
rows = [json.loads(s) for s in (ROOT / 'run_metrics.jsonl').read_text().splitlines()]
episodes = [r for r in rows if r.get('episode/total_reward') is not None]
train = [r for r in rows if r.get('train/loss_q') is not None]
metrics = ['episode/total_reward', 'episode/hazard_person_steps',
           'episode/initial_inside', 'episode/perceptibility',
           'episode/robot_speed_mean', 'episode/mode_share/guide',
           'episode/mode_switch_rate', 'episode/held_clear_success',
           'episode/reentries_after_clear', 'episode/evac_time_100']

def stats(rs, keys):
    out = {'n': len(rs)}
    for k in keys:
        a = np.array([r[k] for r in rs if isinstance(r.get(k), (float, int))])
        if len(a):
            out[k] = {'mean': float(a.mean()), 'sd': float(a.std()),
                      'median': float(np.median(a))}
    return out

blocks = {}
for lo, hi in [(1,500),(501,1000),(1001,1500),(1501,2000),(2001,2500),
               (2501,3000),(3001,4000)]:
    rs = [r for r in episodes if lo <= r['global_episode'] <= hi]
    blocks[f'{lo}-{hi}'] = stats(rs, metrics)

site = {}
for i in sorted({r['episode/site_index'] for r in episodes}):
    site[str(i)] = {
        'early': stats([r for r in episodes if r['episode/site_index']==i
                        and 501 <= r['global_episode'] <=1000], metrics),
        'late': stats([r for r in episodes if r['episode/site_index']==i
                       and r['global_episode'] >3000], metrics)}

# Descriptive OLS, not a causal estimate. Account for known episode variation.
rs = [r for r in episodes if r['global_episode']>500]
features = ['intercept','episode_per_1000','site_1','site_2','initial_inside',
            'hazard_fraction','perceptibility','initial_population']
X = np.array([[1,r['global_episode']/1000,int(r['episode/site_index']==1),
               int(r['episode/site_index']==2),r['episode/initial_inside'],
               r['episode/hazard_area_fraction'],r['episode/perceptibility'],
               r['episode/initial_population']] for r in rs])
y = np.array([r['episode/total_reward'] for r in rs])
b = np.linalg.lstsq(X,y,rcond=None)[0]
err=y-X@b
inv=np.linalg.pinv(X.T@X)
# HC1 robust standard errors, without accounting for temporal correlation.
cov=inv @ ((X*err[:,None]).T @ (X*err[:,None])) @ inv *len(y)/(len(y)-X.shape[1])
se=np.sqrt(np.maximum(0,np.diag(cov)))
reg={'coefficients':dict(zip(features,b.tolist())),
     'standard_errors_HC1':dict(zip(features,se.tolist())),
     'r_squared':float(1-np.sum(err**2)/np.sum((y-y.mean())**2))}
summary={'episode_rows':len(episodes),'train_rows':len(train),
         'last_episode':max(r['global_episode'] for r in episodes),
         'blocks':blocks,'by_site':site,'descriptive_regression':reg,
         'train_early':stats(train[:200],['train/alpha','train/entropy','train/loss_q','train/q_team']),
         'train_late':stats(train[-200:],['train/alpha','train/entropy','train/loss_q','train/q_team']),
         'alpha_floor_fraction':float(np.mean([r['train/alpha']<=.050001 for r in train]))}
(ROOT/'analysis.json').write_text(json.dumps(summary,indent=2))

fig,axs=plt.subplots(3,2,figsize=(12,10),constrained_layout=True)
def plot(ax,rs,key,label,window=100):
    xs=np.array([r['global_episode'] for r in rs]); ys=np.array([r[key] for r in rs])
    ax.scatter(xs,ys,s=2,alpha=.1)
    if len(ys)>=window:
        ax.plot(xs[window-1:],np.convolve(ys,np.ones(window)/window,'valid'),lw=1.5)
    ax.set(xlabel='Episode',ylabel=label);ax.grid(alpha=.2)
plot(axs[0,0],episodes,'episode/total_reward','Return (higher is better)')
plot(axs[0,1],episodes,'episode/hazard_person_steps','Hazard person-steps')
plot(axs[1,0],episodes,'episode/mode_share/guide','Guide fraction')
plot(axs[1,1],episodes,'episode/robot_speed_mean','Actual speed (m/s)')
plot(axs[2,0],train,'train/alpha','Alpha',50)
plot(axs[2,1],train,'train/entropy','Joint action entropy',50)
fig.savefig(ROOT/'learning_curves.png',dpi=150)
print(json.dumps(summary,indent=2))
