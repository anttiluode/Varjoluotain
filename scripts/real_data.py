#!/usr/bin/env python3
"""
Reproduce the real-photo reconstructions of Saunders, Murray-Bruce & Goyal,
Nature 565, 472 (2019), Fig. 4 column c, with this Python port.

The photos are not redistributed here.  Clone their repository first:

    git clone --depth 1 https://github.com/Computational-Periscopy/Ordinary-Camera
    python scripts/real_data.py --data Ordinary-Camera            # published occluder estimates
    python scripts/real_data.py --data Ordinary-Camera --locate   # also locate the occluder ourselves

--locate is experimental: on these photos the funnel converges to a wrong
position (README, section 4), so the published estimates are the default.

Scores are Pearson correlations between reconstruction and the ground-truth
image shown on the hidden screen (all colour channels together).
"""
import argparse, json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from varjoluotain import geometry as G, transport as T, solve as S, realdata as R, locate as L

OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def corr(X, gt):
    a = X.ravel() - X.mean()
    b = gt.astype(float).ravel() - gt.mean()
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def show(X):
    return np.clip(X / max(np.percentile(X, 99.5), 1e-12), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to a clone of Computational-Periscopy/Ordinary-Camera")
    ap.add_argument("--locate", action="store_true", help="locate the occluder from each photo with the funnel")
    ap.add_argument("--iters", type=int, default=500)
    args = ap.parse_args()
    root = R.find_d11(args.data)
    os.makedirs(OUT, exist_ok=True)
    keys = ["bu", "mushroom", "tommy", "rgb"]
    cols = 4 if args.locate else 3
    fig, ax = plt.subplots(len(keys), cols, figsize=(2.9 * cols, 2.5 * len(keys)))
    report = {}
    for r, key in enumerate(keys):
        fname, occ, scale, lam = R.D11_SCENES[key]
        photo, gt, _ = R.load_measurement(os.path.join(root, fname))
        t = time.time()
        A = T.build_A(G.paper_compatible(G.d11(occ)))
        X = S.Reconstructor(A, 126, (29, 36), background="diff").solve(photo, scale, lam, args.iters, nonneg=False)
        rep = {"published_occluder": occ, "corr_published_occluder": round(corr(X, gt), 3),
               "seconds": round(time.time() - t, 1)}
        ax[r, 0].imshow(photo / photo.max()); ax[r, 1].imshow(gt); ax[r, 2].imshow(show(X))
        if args.locate:
            t = time.time()
            found, ledger = L.funnel(G.paper_compatible(G.d11(occ)), photo, log=None)
            A2 = T.build_A(G.paper_compatible(G.d11(tuple(found))))
            X2 = S.Reconstructor(A2, 126, (29, 36), background="diff").solve(photo, scale, lam, args.iters, nonneg=False)
            rep.update(located_occluder=[round(float(v), 4) for v in found],
                       located_minus_published_mm=[round(1000 * (a - b), 1) for a, b in zip(found, occ)],
                       located_minus_tape_mm=[round(1000 * (a - b), 1) for a, b in zip(found, R.D11_TAPE_MEASURED)],
                       corr_located_occluder=round(corr(X2, gt), 3), funnel=ledger,
                       locate_seconds=round(time.time() - t, 1))
            ax[r, 3].imshow(show(X2))
        report[key] = rep
        print(key, {k: v for k, v in rep.items() if k != "funnel"}, flush=True)
    titles = ["photo of the wall", "hidden screen (truth)", "reconstruction\n(published occluder)",
              "reconstruction\n(occluder located here)"]
    for c in range(cols):
        ax[0, c].set_title(titles[c], fontsize=9)
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "real_d11.png"), dpi=110)
    json.dump(report, open(os.path.join(OUT, "real_d11.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
