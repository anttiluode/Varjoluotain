#!/usr/bin/env python3
"""
Find the occluder from the photo alone, with Luotain's escalating funnel,
and show the certificates: the residue a wrong hypothesis leaves on the wall.

    python scripts/locate.py                  # simulated room, known answer
"""
import json, os, sys, time
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from varjoluotain import geometry as G, transport as T, solve as S, camera as C, images as I, locate as L
from skimage.metrics import peak_signal_noise_ratio as psnr

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
TRUTH = np.array((0.475, 0.570, 0.214))


def main():
    os.makedirs(OUT, exist_ok=True)
    room = G.d11(tuple(TRUTH))
    A = T.build_A(room)
    cam = C.Camera(photons=1e6)
    x = I.sample("astronaut")
    Y, sig, k = C.photograph(A, x, 126, cam)

    t = time.time()
    found, ledger = L.funnel(room, Y, truth=TRUTH)
    secs = time.time() - t
    err_mm = (found - TRUTH) * 1000

    # what exhaustive search would cost at the last level's resolution and model
    (ua, ub), (va, vb), (ya, yb) = L.search_box(room)
    fine = L.LEVELS[-1][1]
    n_fine = np.prod([int(round((b - a) / st)) + 1 for (a, b), st in zip(((ua, ub), (va, vb), (ya, yb)), fine)])
    brute_s = n_fine * ledger[-1]["ms_per_hypothesis"] / 1000

    # certificates: the residue each hypothesis leaves on the wall
    wrong = L.corner_from_uvy(room, L.uvy_from_corner(room, TRUTH) + np.array([0.0, 0.0, -0.06]))   # 6 cm closer to the screen
    shift = TRUTH + np.array([0.02, 0.0, 0.0])                                                        # 2 cm sideways
    maps = {"hypothesis: 6 cm nearer the screen": L.residue_map(room, Y, wrong),
            "hypothesis: 2 cm to the side": L.residue_map(room, Y, shift),
            "located occluder": L.residue_map(room, Y, found)}
    rec_room = replace(room, occluders=G.square_on_stand(*found))
    Af = T.build_A(rec_room)
    rec = S.Reconstructor(Af, 126, (29, 36))
    sc = C.scales(Af, cam, k)
    X = np.clip(rec.solve(Y, sc, rec.discrepancy_lambda(Y, sig, sc), iters=400), 0, 1)

    vmax = max(m.max() for m in maps.values())
    fig, ax = plt.subplots(1, 5, figsize=(15, 3.3))
    ax[0].imshow(Y / Y.max()); ax[0].set_title("the photo", fontsize=9)
    for a, (t_, m) in zip(ax[1:4], maps.items()):
        a.imshow(m, cmap="magma", vmin=0, vmax=vmax)
        a.set_title("residue left by\n" + t_, fontsize=9)
        a.set_xlabel("rms %.0f e-" % np.sqrt((m ** 2).mean()), fontsize=8)
    rec_psnr = float(psnr(x, X, data_range=1))
    ax[4].imshow(X); ax[4].set_title("reconstruction with the\nlocated occluder", fontsize=9)
    ax[4].set_xlabel("PSNR %.1f dB" % rec_psnr, fontsize=8)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "locate.png"), dpi=110)
    report = {"truth_corner": TRUTH.tolist(), "found_corner": [round(float(v), 4) for v in found],
              "error_mm": [round(float(v), 1) for v in err_mm], "funnel_seconds": round(secs, 1),
              "hypotheses_tested": int(sum(r["tested"] for r in ledger)),
              "exhaustive_at_final_resolution": {"hypotheses": int(n_fine), "estimated_days": round(brute_s / 86400, 1)},
              "residue_rms": {k_: round(float(np.sqrt((m ** 2).mean())), 1) for k_, m in maps.items()},
              "noise_sigma_electrons": [round(float(v), 1) for v in sig],
              "reconstruction_psnr_with_located_occluder": round(rec_psnr, 2),
              "funnel": ledger}
    json.dump(report, open(os.path.join(OUT, "locate.json"), "w"), indent=1)
    print(json.dumps({k_: v for k_, v in report.items() if k_ != "funnel"}, indent=1))


if __name__ == "__main__":
    main()
