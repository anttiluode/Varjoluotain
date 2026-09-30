#!/usr/bin/env python3
"""
The ledger: what one photo of the wall can decide about the hidden screen,
with and without the occluder.  Luotain's three piles, for images:

  observable modes  - scene changes that move the photo by more than the noise
  twins             - scene changes that do not; two hidden screens that differ
                      only there give photos no camera can tell apart
  observability map - per screen patch, how much of what we know comes from
                      the wall rather than from the prior

    python scripts/ledger.py
"""
import json, os, sys
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from varjoluotain import geometry as G, transport as T, solve as S, camera as C, images as I, ledger as Lg

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
TAPE = (0.475, 0.570, 0.214)


def main():
    os.makedirs(OUT, exist_ok=True)
    room = G.d11(TAPE)
    cam = C.Camera(photons=1e6)
    k = C.exposure(T.build_A(replace(room, occluders=[])), cam)    # one camera exposure for both rooms
    report = {}
    x1, target = I.sample("astronaut"), I.sample("coffee")
    panels = {}
    for label, s in (("with occluder", room), ("no occluder", replace(room, occluders=[]))):
        A = T.build_A(s)
        At = S.background_free(A, 126)
        sv, V = Lg.modes(At)
        _, sig, _ = C.photograph(A, x1, 126, cam, k)
        sigma = float(sig.mean())
        obs = Lg.observable(sv, k, sigma)
        # twin: keep what the wall sees of the astronaut, take the rest from the coffee cup
        x2 = np.stack([Lg.twin(At, x1[:, :, c].ravel(), target[:, :, c].ravel(), k, sigma, V, sv)[0]
                       for c in range(3)], -1).reshape(x1.shape)
        d_photo = np.linalg.norm(np.stack([k * At @ (x2 - x1)[:, :, c].ravel() for c in range(3)]))
        noise = sigma * np.sqrt(3 * At.shape[0])
        omap = Lg.observability_map(At, k, sigma, (29, 36))
        vis = {n: round(Lg.visible_fraction(sv, V, I.sample(n).mean(2).ravel(), k, sigma), 3)
               for n in ("astronaut", "coffee", "chelsea", "rocket")}
        report[label] = {"observable_modes": int(obs.sum()), "of_modes": int(len(sv)),
                         "twin_photo_difference_over_noise": round(float(d_photo / noise), 4),
                         "twin_pixels_outside_0_1": round(float(((x2 < 0) | (x2 > 1)).mean()), 3),
                         "mean_observability": round(float(omap.mean()), 3),
                         "share_of_scene_energy_the_photo_decides": vis}
        panels[label] = (k * sv / sigma, x2, omap)
        print(label, report[label], flush=True)

    # photos of the wall, no plate: the astronaut and its twin (noise-free, same scale)
    A0 = T.build_A(replace(room, occluders=[]))
    amb = C.room_light(126, cam)
    shot = lambda x: np.stack([(k * A0 @ x[:, :, c].ravel()).reshape(126, 126) + amb for c in range(3)], -1)
    twin0 = panels["no occluder"][1]
    p1, p2 = shot(x1), shot(twin0)
    top = max(p1.max(), p2.max())

    fig, ax = plt.subplots(2, 4, figsize=(13, 6.4))
    a = ax[0, 0]
    for label, c in (("with the plate", "C0"), ("no plate", "C3")):
        key = "with occluder" if label == "with the plate" else "no occluder"
        a.semilogy(np.maximum(panels[key][0], 1e-8), c, label="%s: %d observable" % (label, report[key]["observable_modes"]))
    a.axhline(1, color="k", lw=0.8, ls="--")
    a.set_ylim(1e-8, None)
    a.set_title("photo change per unit screen change,\nin units of camera noise", fontsize=9)
    a.set_xlabel("screen pattern (mode), strongest first", fontsize=8); a.legend(fontsize=7, loc="lower left")
    tiles = [(ax[0, 1], x1, "screen shows A"),
             (ax[0, 2], np.clip(twin0, 0, 1), "a twin of A, no plate"),
             (ax[0, 3], np.clip(panels["with occluder"][1], 0, 1), "the same attempt, plate present"),
             (ax[1, 0], p1 / top, "wall photo of A, no plate"),
             (ax[1, 1], p2 / top, "wall photo of the twin, no plate")]
    for a, im, t in tiles:
        a.imshow(im); a.set_title(t, fontsize=9); a.set_xticks([]); a.set_yticks([])
    ax[1, 1].set_xlabel("differs from the left by %.3f x noise" % report["no occluder"]["twin_photo_difference_over_noise"], fontsize=8)
    for a, key, t in ((ax[1, 2], "with occluder", "what the wall decides, plate"), (ax[1, 3], "no occluder", "what the wall decides, no plate")):
        im = a.imshow(panels[key][2], vmin=0, vmax=1, cmap="viridis")
        a.set_title(t, fontsize=9); a.set_xticks([]); a.set_yticks([])
        a.set_xlabel("mean %.2f (1 = data, 0 = prior)" % report[key]["mean_observability"], fontsize=8)
    fig.colorbar(im, ax=ax[1, 3], fraction=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "ledger.png"), dpi=110)
    json.dump(report, open(os.path.join(OUT, "ledger.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
