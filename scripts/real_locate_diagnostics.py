#!/usr/bin/env python3
"""
Why locating the occluder from a real photo is hard.

For the paper's BU and RGB-bar photos, move the occluder away from the
published estimate and record, for each position:
  - the reconstruction objective (data misfit + TV), which is what the photo
    alone can use to judge a position, and
  - the correlation of the reconstruction with the true screen, which the
    photo alone cannot see (diagnostic only).

    python scripts/real_locate_diagnostics.py --data Ordinary-Camera
"""
import argparse, json, os, sys
from dataclasses import replace
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from varjoluotain import geometry as G, transport as T, solve as S, realdata as R

OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def corr(X, gt):
    a = X.ravel() - X.mean()
    b = gt.astype(float).ravel() - gt.mean()
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    root = R.find_d11(args.data)
    report = {}
    for key in ("bu", "rgb"):
        fname, occ, scale, lam = R.D11_SCENES[key]
        photo, gt, _ = R.load_measurement(os.path.join(root, fname))
        base = G.paper_compatible(G.d11(occ))
        moves = {"published estimate": occ, "tape measure": R.D11_TAPE_MEASURED,
                 "2 cm left": (occ[0] - 0.02, occ[1], occ[2]), "2 cm right": (occ[0] + 0.02, occ[1], occ[2]),
                 "2 cm higher": (occ[0], occ[1], occ[2] + 0.02), "3 cm nearer the wall": (occ[0], occ[1] + 0.03, occ[2])}
        rows = {}
        for label, c in moves.items():
            A = T.build_A(replace(base, occluders=G.square_on_stand(*c)))
            rec = S.Reconstructor(A, 126, (29, 36), background="diff")
            X = rec.solve(photo, scale, lam, 300, nonneg=False)
            ys = rec.prepare(photo)
            mis = sum(0.5 * np.linalg.norm(scale[ch] * rec.At @ X[:, :, ch].ravel() - ys[:, ch]) ** 2 for ch in range(3))
            tvp = sum(lam[ch] * S.tv(X[:, :, ch]) for ch in range(3))
            rows[label] = {"corner": [round(float(v), 4) for v in c], "objective": float(mis + tvp),
                           "misfit": float(mis), "corr_with_truth": round(corr(X, gt), 3)}
        ref = rows["published estimate"]["objective"]
        for r in rows.values():
            r["objective_vs_published"] = round(r["objective"] / ref, 4)
        report[key] = rows
        for label, r in rows.items():
            print("%-4s %-22s objective x%.3f   corr with truth %.3f" % (key, label, r["objective_vs_published"], r["corr_with_truth"]), flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(report, open(os.path.join(OUT, "real_locate_diagnostics.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
