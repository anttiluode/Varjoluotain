"""
Locating the occluder: Luotain's probe -> residue funnel.

Every candidate occluder position is a hypothesis about the room.  A
hypothesis is refuted when no non-negative hidden scene can explain the photo
under it: the shadow edges it predicts are not where the photo has them.  The
misfit left over (its residue map) is the certificate.

Hypotheses are parametrised the way shadows are made: by the direction (u, v)
from the centre of the hidden screen to the occluder, and its depth y.  Moving
an occluder along that direction barely moves its shadow; that is the long,
narrow valley every localisation has to walk down, so the grid follows it.

Checking a hypothesis at full resolution costs about a second, and the search
box holds thousands.  So the funnel escalates exactly like Luotain's
2-element -> 3-element -> prover ladder: a cheap model tests the whole coarse
grid; only the cells it cannot refute are split into finer children and
tested by a better model; and so on.
"""
import time
from dataclasses import replace
import numpy as np
from scipy.optimize import nnls
from . import transport as T
from .geometry import Setup, square_on_stand
from .solve import background_free, photo_columns

PLATE = 0.075


def model_at(s: Setup, rows: int, cols: int, wall_bin: int, sub: int) -> Setup:
    """A cheaper model of the same room: fewer, larger scene patches tiling the
    same lit area, the wall sampled at the centres of wall_bin x wall_bin photo
    bins, fewer point sources per patch."""
    if (rows, cols, wall_bin, sub) == (s.rows, s.cols, 1, s.sub):
        return s
    n = s.wall_n // wall_bin
    step = s.wall_w / (s.wall_n - 1)
    off = (wall_bin - 1) / 2 * step
    if (rows, cols) == (s.rows, s.cols):
        scene = {}
    else:
        x0, x1, z0, z1 = s.screen_extent()
        bw, bh = (x1 - x0) / cols, (z1 - z0) / rows
        scene = dict(rows=rows, cols=cols, block_w=bw, block_h=bh, anchor_mode="tile",
                     mon_x0=x0 + bw, mon_z0=z0 + bh)
    return replace(s, wall_n=n, wall_x0=s.wall_x0 + off, wall_z0=s.wall_z0 + off,
                   wall_w=(n - 1) * wall_bin * step, wall_h=(n - 1) * wall_bin * step,
                   sub=sub, visibility_convention="center", **scene)


def bin_photo(Y: np.ndarray, f: int) -> np.ndarray:
    if f == 1:
        return Y
    n = Y.shape[0] // f
    return Y[:n * f, :n * f].reshape(n, f, n, f, -1).mean((1, 3))


def _nn_fit(At, ys, iters=300):
    """Non-negative least squares for every channel; returns fitted values."""
    if At.shape[1] <= 200:
        return np.stack([At @ nnls(At, ys[:, c])[0] for c in range(ys.shape[1])], 1)
    G = At.T @ At
    L = float(np.linalg.eigvalsh(G)[-1])
    B = At.T @ ys
    X = np.zeros_like(B); Yv = X.copy(); t = 1.0
    for _ in range(iters):
        Xn = np.maximum(Yv - (G @ Yv - B) / L, 0)
        tn = (1 + np.sqrt(1 + 4 * t * t)) / 2
        Yv = Xn + (t - 1) / tn * (Xn - X)
        X, t = Xn, tn
    return At @ X


def misfit(A: np.ndarray, Y: np.ndarray, n: int) -> float:
    """Squared residue left by the best non-negative scene (each channel free)."""
    At = background_free(A, n, "poly", 1)
    ys = background_free(photo_columns(Y), n, "poly", 1)
    return float(((_nn_fit(At, ys) - ys) ** 2).sum())


def residue_map(s: Setup, Y: np.ndarray, corner, bin_f: int = 2) -> np.ndarray:
    """Where a hypothesis fails: |photo - best explanation| per wall pixel."""
    m = model_at(s, s.rows, s.cols, bin_f, 4)
    A = T.build_A(replace(m, occluders=square_on_stand(*corner)))
    Yb = bin_photo(Y, bin_f)
    At = background_free(A, m.wall_n, "poly", 1)
    ys = background_free(photo_columns(Yb), m.wall_n, "poly", 1)
    return np.sqrt(((_nn_fit(At, ys) - ys) ** 2).sum(1)).reshape(m.wall_n, m.wall_n)


# ---------------------------------------------------------------- geometry --
def screen_centre(s: Setup):
    x0, x1, z0, z1 = s.screen_extent()
    return np.array([(x0 + x1) / 2, 0.0, (z0 + z1) / 2])


def corner_from_uvy(s: Setup, h) -> np.ndarray:
    """(u, v, y) -> the plate's lower-left corner."""
    c = screen_centre(s)
    u, v, y = h
    return np.array([c[0] + u * y - PLATE / 2, y, c[2] + v * y - PLATE / 2])


def uvy_from_corner(s: Setup, corner) -> np.ndarray:
    c = screen_centre(s)
    x, y, z = corner
    return np.array([(x + PLATE / 2 - c[0]) / y, y, (z + PLATE / 2 - c[2]) / y])[[0, 2, 1]]


def search_box(s: Setup, y=(0.40, 0.76)):
    """Directions whose shadow centre lands in the camera's view, and a depth range."""
    c = screen_centre(s)
    u = ((s.wall_x0 - c[0]) / s.D, (s.wall_x0 + s.wall_w - c[0]) / s.D)
    v = ((s.wall_z0 - c[2]) / s.D, (s.wall_z0 + s.wall_h - c[2]) / s.D)
    return u, v, y


# ------------------------------------------------------------------ funnel --
LEVELS = (
    # cheaper model of the room (rows, cols, wall_bin, sub), grid step (du, dv, dy), keep margin, keep cap
    ((6, 8, 6, 2),    (0.02, 0.02, 0.03),      1.0, 400),
    ((6, 8, 3, 3),    (0.01, 0.01, 0.015),     0.5, 120),
    ((12, 15, 3, 3),  (0.005, 0.005, 0.0075),  0.25, 6),
    ((29, 36, 2, 4),  (0.0025, 0.0025, 0.004), 0.0, 1),
)


def _children(parents: np.ndarray, step) -> np.ndarray:
    offs = np.stack(np.meshgrid(*[np.array([-1, 0, 1])] * 3, indexing="ij"), -1).reshape(-1, 3)
    kids = (parents[:, None, :] + offs[None] * np.asarray(step)).reshape(-1, 3)
    key = np.round(kids / (np.asarray(step) / 4)).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return kids[np.sort(idx)]


def funnel(s: Setup, Y: np.ndarray, levels=LEVELS, box=None, truth=None, log=print):
    """Locate the occluder. Returns (plate corner, ledger rows).
    truth (a plate corner) is only used to report whether it was ever refuted."""
    u, v, y = box or search_box(s)
    tu = None if truth is None else uvy_from_corner(s, truth)
    ledger, alive = [], None
    for k, (mdl, step, margin, cap) in enumerate(levels):
        t0 = time.time()
        if alive is None:
            ax = [np.arange(a, b + 1e-9, st) for (a, b), st in zip((u, v, y), step)]
            H = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
        else:
            H = _children(alive, step)
        m = model_at(s, *mdl)
        Yb = bin_photo(Y, mdl[2])
        cost = np.array([misfit(T.build_A(replace(m, occluders=square_on_stand(*corner_from_uvy(s, h)))),
                                Yb, m.wall_n) for h in H])
        order = np.argsort(cost)
        keep = [i for i in order if cost[i] <= cost[order[0]] * (1 + margin)][:cap]
        alive = H[keep]
        row = {"level": k + 1,
               "model": "%dx%d patches, %dx%d wall, %dx%d sources" % (mdl[0], mdl[1], m.wall_n, m.wall_n, mdl[3], mdl[3]),
               "step_u_v_y": list(step), "tested": int(len(H)), "kept": int(len(keep)),
               "refuted": int(len(H) - len(keep)), "seconds": round(time.time() - t0, 1),
               "ms_per_hypothesis": round((time.time() - t0) / len(H) * 1000, 1),
               "best_corner": [round(float(x), 4) for x in corner_from_uvy(s, alive[0])]}
        if tu is not None:
            covers = np.all(np.abs(alive - tu) <= np.asarray(step) * 1.0001, axis=1)
            row["truth_still_covered"] = bool(covers.any())
        ledger.append(row)
        if log:
            log("level %d  %-38s tested %6d  kept %4d  refuted %6d  %6.1fs  best %s%s" % (
                row["level"], row["model"], row["tested"], row["kept"], row["refuted"], row["seconds"],
                row["best_corner"], "" if tu is None else "  truth covered: %s" % row["truth_still_covered"]))
    return corner_from_uvy(s, alive[0]), ledger
