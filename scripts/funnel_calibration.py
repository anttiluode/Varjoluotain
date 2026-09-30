#!/usr/bin/env python3
"""
Calibrate the occluder funnel on simulated rooms with a known answer, the way
Luotain first ran its probes against the finished ETP map: does the true
position ever get refuted, and where does the funnel end?

    python scripts/funnel_calibration.py astronaut coffee rocket

Writes results/funnel_calibration_<scenes>.json.
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from varjoluotain import geometry as G, transport as T, camera as C, images as I, locate as L

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
TRUTH = np.array((0.475, 0.570, 0.214))


def main(names):
    room = G.d11(tuple(TRUTH))
    A = T.build_A(room)
    out = {"truth_corner": TRUTH.tolist(), "photons": 1e6, "scenes": {}}
    for name in names:
        Y, _, _ = C.photograph(A, I.sample(name), 126, C.Camera(photons=1e6))
        print("==", name, flush=True)
        t = time.time()
        best, ledger = L.funnel(room, Y, truth=TRUTH, log=lambda m: print(m, flush=True))
        err = (best - TRUTH) * 1000
        print("final corner", np.round(best, 4), "error mm", np.round(err, 1), "%.0fs" % (time.time() - t), flush=True)
        out["scenes"][name] = {"ledger": ledger, "final": best.tolist(), "error_mm": err.tolist()}
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "funnel_calibration_%s.json" % "_".join(names)), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or ["astronaut"])
