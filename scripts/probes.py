#!/usr/bin/env python3
"""
The occluder is the probe.  Luotain found that ten structured 2-element tables
settle 90% of what thousands of random tables cannot.  Which occluder makes
the hidden screen most decidable from a photo of the wall?

Same room, same camera exposure, same position (the paper's plate centre at
57 cm from the screen); only the occluder changes.

    python scripts/probes.py
"""
import json, os, sys, time
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from varjoluotain import geometry as G, transport as T, solve as S, camera as C, images as I, ledger as Lg
from varjoluotain.geometry import Rect, MaskOccluder

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
TAPE = (0.475, 0.570, 0.214)
CX, CZ, Y = TAPE[0] + 0.0375, TAPE[2] + 0.0375, TAPE[1]


def square(size):
    return Rect(CX - size / 2, CX + size / 2, CZ - size / 2, CZ + size / 2, Y)


def designs():
    rng = np.random.default_rng(0)
    rand = rng.random((8, 8)) < 0.5
    hole = np.ones((40, 40), bool); hole[19:21, 19:21] = False
    return [
        ("no occluder", [], []),
        ("paper: 7.5 cm plate on a stand", G.square_on_stand(*TAPE), []),
        ("7.5 cm plate, floating", [square(0.075)], []),
        ("the stand alone", [G.square_on_stand(*TAPE)[1]], []),
        ("2 cm speck", [square(0.02)], []),
        ("15 cm plate", [square(0.15)], []),
        ("15 cm random mask (8x8, half open)", [], [MaskOccluder(CX - 0.075, CZ - 0.075, 0.15, 0.15, Y, rand)]),
        ("vertical edge (door frame)", [Rect(CX, CX + 3.0, -1.0, 3.0, Y)], []),
        ("2 cm pinhole in a 40 cm plate", [], [MaskOccluder(CX - 0.2, CZ - 0.2, 0.4, 0.4, Y, hole)]),
    ]


def main():
    os.makedirs(OUT, exist_ok=True)
    base = G.d11(TAPE)
    cam = C.Camera(photons=1e6)
    k_ref = C.exposure(T.build_A(replace(base, occluders=[])), cam)   # one camera exposure for every room
    names = ("astronaut", "coffee", "chelsea", "rocket")
    xs = [I.sample(n) for n in names]
    rows, report = [], {}
    for label, rects, masks in designs():
        t = time.time()
        A = T.build_A(replace(base, occluders=rects, masks=masks))
        rec = S.Reconstructor(A, 126, (29, 36))
        sv, V = Lg.modes(rec.At)
        ps, ss, recs, sigs = [], [], [], []
        for x in xs:
            Y_, sig, _ = C.photograph(A, x, 126, cam, k=k_ref)
            sc = C.scales(A, cam, k=k_ref)
            lam = rec.discrepancy_lambda(Y_, sig, sc)
            X = np.clip(rec.solve(Y_, sc, lam, iters=400), 0, 1)
            ps.append(psnr(x, X, data_range=1)); ss.append(ssim(x, X, channel_axis=2, data_range=1))
            recs.append(X); sigs.append(sig.mean())
        obs = int(Lg.observable(sv, k_ref, float(np.mean(sigs))).sum())
        report[label] = {"observable_modes": obs, "mean_psnr": round(float(np.mean(ps)), 2),
                         "mean_ssim": round(float(np.mean(ss)), 3), "seconds": round(time.time() - t, 1)}
        rows.append((label, recs[0]))
        print(label, report[label], flush=True)
    np.savez_compressed(os.path.join(OUT, "probes_reconstructions.npz"),
                        **{"r%d" % i: X for i, (_, X) in enumerate(rows)}, labels=np.array([l for l, _ in rows]))
    fig, ax = plt.subplots(3, 3, figsize=(9.5, 8.4))
    for a, (label, X) in zip(ax.ravel(), rows):
        a.imshow(X); a.set_xticks([]); a.set_yticks([])
        a.set_title(label, fontsize=9)
        a.set_xlabel("%d observable modes, %.1f dB" % (report[label]["observable_modes"], report[label]["mean_psnr"]), fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "probes.png"), dpi=110)
    json.dump(report, open(os.path.join(OUT, "probes.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
