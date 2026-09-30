"""
The ledger: what a photo of the wall can and cannot decide about the hidden scene.

Luotain keeps three piles: implications a probe refutes, implications a proof
settles, and the residue nobody has decided.  The same bookkeeping applies
here, with the transport matrix as the probe set:

* modes of the hidden scene that move the photo by more than the camera noise
  are decided by the photo (the observable part);
* modes that move it by less are "fingerprint twins": two hidden scenes that
  differ along them give photos no camera can tell apart.  Whatever a
  reconstruction shows there comes from the prior (TV), not from the wall.

All quantities are for the background-cancelled matrix A~ (room light unknown).
"""
import numpy as np


def modes(At: np.ndarray):
    """Singular values (descending) and right singular vectors of A~."""
    w, V = np.linalg.eigh(At.T @ At)
    w = np.clip(w[::-1], 0, None)
    return np.sqrt(w), V[:, ::-1]


def observable(sv: np.ndarray, k: float, sigma: float) -> np.ndarray:
    """A unit-norm scene change along mode i moves the photo by k * sv_i (electrons).
    It is observable when that exceeds one noise standard deviation."""
    return k * sv >= sigma


def twin(At, x1, target, k, sigma, V=None, sv=None):
    """Keep what the photo sees of x1, take everything it cannot see from `target`.
    Returns (x2, photo difference norm, noise norm) for one channel (flat vectors)."""
    if V is None:
        sv, V = modes(At)
    inv = ~observable(sv, k, sigma)
    Vi = V[:, inv]
    x2 = x1 + Vi @ (Vi.T @ (target - x1))
    dphoto = np.linalg.norm(k * At @ (x2 - x1))
    return x2, float(dphoto), float(sigma * np.sqrt(At.shape[0]))


def observability_map(At, k, sigma, shape, prior_std=0.25):
    """Per patch: share of its posterior certainty that comes from the photo
    (diagonal of the resolution matrix (F + P)^-1 F, F = Fisher information,
    P = prior precision).  1 = the wall decides the patch, 0 = only the prior."""
    F = (k / sigma) ** 2 * (At.T @ At)
    R = np.linalg.solve(F + np.eye(F.shape[0]) / prior_std ** 2, F)
    return np.diag(R).reshape(shape)


def visible_fraction(sv, V, x, k, sigma):
    """Share of a scene's (mean-removed) energy that lies in observable modes."""
    x = x - x.mean()
    c = V.T @ x
    vis = observable(sv, k, sigma)
    return float((c[vis] ** 2).sum() / (c ** 2).sum())
