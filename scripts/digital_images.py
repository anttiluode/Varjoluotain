#!/usr/bin/env python3
"""
Computational periscopy on ordinary digital images.

A digital image is shown on a screen the camera cannot see.  The camera
photographs a blank wall; a small occluder between screen and wall casts a
penumbra.  From that one photo we reconstruct the image, and compare with the
same room without the occluder.

    python scripts/digital_images.py
"""
import json, os, sys, time
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from varjoluotain import geometry as G, transport as T, solve as S, camera as C, images as I

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
TAPE = (0.475, 0.570, 0.214)          # the paper's hand-measured occluder corner


def reconstruct(A, rec, Y, sig, cam, k):
    sc = C.scales(A, cam, k)
    lam = rec.discrepancy_lambda(Y, sig, sc)
    return np.clip(rec.solve(Y, sc, lam, iters=400), 0, 1), lam


def scores(x, X):
    return round(float(psnr(x, X, data_range=1)), 2), round(float(ssim(x, X, channel_axis=2, data_range=1)), 3)


def main():
    os.makedirs(OUT, exist_ok=True)
    room = G.d11(TAPE)
    empty = replace(room, occluders=[])
    t = time.time()
    A, A0 = T.build_A(room), T.build_A(empty)
    rec, rec0 = S.Reconstructor(A, 126, (29, 36)), S.Reconstructor(A0, 126, (29, 36))
    print("transport matrices %.0fs" % (time.time() - t))
    cam = C.Camera(photons=1e6)
    k = C.exposure(A0, cam)             # one camera exposure for both rooms
    scenes = [(n, I.sample(n)) for n in ("astronaut", "coffee", "chelsea", "rocket", "colorwheel")]
    scenes.append(("VARJO", I.text("VARJO")))
    fig, ax = plt.subplots(len(scenes), 4, figsize=(11, 2.3 * len(scenes)))
    report = {"camera": vars(cam), "occluder_corner": TAPE, "scenes": {}}
    for r, (name, x) in enumerate(scenes):
        Y, sig, _ = C.photograph(A, x, 126, cam, k)
        X, lam = reconstruct(A, rec, Y, sig, cam, k)
        Y0, sig0, _ = C.photograph(A0, x, 126, cam, k)
        X0, _ = reconstruct(A0, rec0, Y0, sig0, cam, k)
        report["scenes"][name] = {"with_occluder_psnr_ssim": scores(x, X), "without_occluder_psnr_ssim": scores(x, X0)}
        print(name, report["scenes"][name], flush=True)
        ax[r, 0].imshow(x); ax[r, 1].imshow(Y / Y.max()); ax[r, 2].imshow(X); ax[r, 3].imshow(X0)
        ax[r, 2].set_xlabel("PSNR %.1f dB, SSIM %.2f" % scores(x, X), fontsize=8)
        ax[r, 3].set_xlabel("PSNR %.1f dB, SSIM %.2f" % scores(x, X0), fontsize=8)
    for c, t_ in enumerate(["hidden screen", "photo of the wall", "reconstructed from the photo",
                            "same room, no occluder"]):
        ax[0, c].set_title(t_, fontsize=10)
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "digital_images.png"), dpi=110)

    # how the result depends on light: one scene, a range of photon budgets
    x = I.sample("astronaut")
    sweep = {}
    for ph in (1e4, 1e5, 1e6, 1e7):
        c2 = replace(cam, photons=ph)
        k2 = C.exposure(A0, c2)
        Y, sig, _ = C.photograph(A, x, 126, c2, k2)
        X, _ = reconstruct(A, rec, Y, sig, c2, k2)
        sweep["%.0e" % ph] = scores(x, X)
    report["astronaut_photon_sweep_psnr_ssim"] = sweep
    print("photon sweep", sweep)
    json.dump(report, open(os.path.join(OUT, "digital_images.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
