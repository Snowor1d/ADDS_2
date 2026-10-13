"""Can followers be read from what the actor and critic already see?

Uses results/fol/*.npz from follower_decode.py (one file per validation level).
For each feature set, a decoder is trained on three levels and tested on the
fourth (held-out level), and, for reference, with a random 5-fold split that
mixes levels (same levels).

  proxy   people within 3.5 / 5 / 10 m of the robot, its mode, inside-hazard
          flag and two interactions: all computable from the robot's ego crop
          and its state vector, so already in both networks' inputs
  actor   the trained actor's 64-d backbone features
  critic  the trained critic's action-independent features (vision + state)
"""
import glob
import os

# Keep BLAS small: this runs next to training jobs.
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "2")

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

HERE = os.path.dirname(os.path.abspath(__file__))
D = [dict(np.load(f, allow_pickle=True))
     for f in sorted(glob.glob(os.path.join(HERE, "results/fol/*.npz")))]


def proxy(d):
    return np.stack([d["n3"], d["n5"], d["n10"], d["mode_guide"], d["inside"],
                     d["n10"] * d["mode_guide"], d["n5"] * d["mode_guide"]],
                    1).astype(float)


FEATS = {"proxy": proxy, "actor": lambda d: d["actor_feat"],
         "critic": lambda d: d["critic_feat"]}


def fit_predict(Xtr, ytr, Xte, model, pca):
    Xtr, Xte = Xtr.astype(np.float64), Xte.astype(np.float64)
    if model == "gbm":
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05
                                             ).fit(Xtr, ytr).predict(Xte)
    keep = Xtr.std(0) > 1e-3                 # drop constant features
    mu, sd = Xtr[:, keep].mean(0), Xtr[:, keep].std(0)
    Ztr = (Xtr[:, keep] - mu) / sd
    Zte = np.clip((Xte[:, keep] - mu) / sd, -10, 10)
    if pca:
        p = PCA(n_components=pca).fit(Ztr)
        Ztr, Zte = p.transform(Ztr), p.transform(Zte)
    return RidgeCV(alphas=np.logspace(-1, 5, 13)).fit(Ztr, ytr).predict(Zte)


def held_out_level(kind, target, model, pca):
    yt, yp = [], []
    for i in range(len(D)):
        tr = [d for j, d in enumerate(D) if j != i]
        Xtr = np.concatenate([FEATS[kind](d) for d in tr])
        ytr = np.concatenate([d[target] for d in tr])
        yt.append(D[i][target])
        yp.append(fit_predict(Xtr, ytr, FEATS[kind](D[i]), model, pca))
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    return r2_score(yt, yp), np.abs(yt - yp).mean()


def same_levels(kind, target, model, pca):
    X = np.concatenate([FEATS[kind](d) for d in D])
    y = np.concatenate([d[target] for d in D])
    p = np.zeros_like(y, dtype=float)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        p[te] = fit_predict(X[tr], y[tr], X[te], model, pca)
    return r2_score(y, p)


if __name__ == "__main__":
    allf = np.concatenate([d["n_follow"] for d in D])
    print("levels:", [str(d["scenario"]) for d in D])
    print("followers per robot-decision: mean %.1f, sd %.1f, share > 0 %.2f, max %d"
          % (allf.mean(), allf.std(), (allf > 0).mean(), allf.max()))
    for target in ("n_follow", "n_follow_in"):
        print(f"--- {target}")
        for kind, model, pca in (("proxy", "ridge", None), ("proxy", "gbm", None),
                                 ("actor", "ridge", None), ("actor", "ridge", 32),
                                 ("critic", "ridge", None), ("critic", "ridge", 32)):
            r2, mae = held_out_level(kind, target, model, pca)
            r2s = same_levels(kind, target, model, pca)
            tag = f"{kind}/{model}" + (f"/pca{pca}" if pca else "")
            print(f"  {tag:18s} held-out level R2 {r2:+.2f} (MAE {mae:.1f})"
                  f" | same levels R2 {r2s:+.2f}")
