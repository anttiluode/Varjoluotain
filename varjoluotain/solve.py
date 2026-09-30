"""
Reconstruction: undo the light transport with a total-variation prior.

    minimise  0.5 || s * A~ x - y~ ||^2 + lam * TV(x)      (per colour channel)

A~, y~ are the transport matrix and measurement with the unknown background
removed, either by the published first-difference trick ("diff", which
cancels a constant) or by projecting out a smooth ramp ("poly", which also
cancels a linear tilt of room light).  FISTA with a Beck-Teboulle TV prox.
"""
import numpy as np


# ------------------------------------------------------------- background --
def c_to_f_rows(M: np.ndarray, n: int) -> np.ndarray:
    """Reorder rows from C order of an n x n image to Fortran (column-major)."""
    rest = M.shape[1:]
    return M.reshape((n, n) + rest).swapaxes(0, 1).reshape((n * n,) + rest)


def diff_op(M: np.ndarray) -> np.ndarray:
    """The published first-difference operator on a stacked vector/matrix."""
    out = np.empty_like(M)
    out[:-1] = M[:-1] - M[1:]
    out[-1] = M[-1]
    return out


def ramp_basis(n: int, order: int = 1) -> np.ndarray:
    u = np.linspace(-1, 1, n)
    X, Z = np.meshgrid(u, -u)           # row 0 = top
    cols = [np.ones(n * n)]
    if order >= 1:
        cols += [X.ravel(), Z.ravel()]
    if order >= 2:
        cols += [(X * X).ravel(), (Z * Z).ravel(), (X * Z).ravel()]
    return np.stack(cols, 1)


def background_free(M: np.ndarray, n: int, mode: str = "poly", order: int = 1) -> np.ndarray:
    """Apply the background-cancelling operator to the rows of M (wall pixels in
    C order of an n x n image, row 0 = top).  Works for A and for photos alike."""
    if mode == "diff":
        return diff_op(c_to_f_rows(M, n))
    if mode == "poly":
        Q, _ = np.linalg.qr(ramp_basis(n, order))
        return M - Q @ (Q.T @ M)
    if mode == "none":
        return M
    raise ValueError(mode)


def noise_gain(mode: str) -> float:
    """How the operator scales white noise (per entry)."""
    return np.sqrt(2.0) if mode == "diff" else 1.0


def photo_columns(Y: np.ndarray) -> np.ndarray:
    return np.stack([Y[:, :, c].ravel() for c in range(Y.shape[2])], 1)


# -------------------------------------------------------------------- TV --
def tv(x: np.ndarray) -> float:
    gx = np.zeros_like(x); gz = np.zeros_like(x)
    gz[:-1] = x[1:] - x[:-1]
    gx[:, :-1] = x[:, 1:] - x[:, :-1]
    return float(np.sqrt(gx ** 2 + gz ** 2).sum())


def _Lt(x):
    p = np.zeros_like(x); q = np.zeros_like(x)
    p[:-1] = x[:-1] - x[1:]
    q[:, :-1] = x[:, :-1] - x[:, 1:]
    return p, q


def _L(p, q):
    I = p + q
    I[1:] -= p[:-1]
    I[:, 1:] -= q[:, :-1]
    return I


def prox_tv(b, lam, iters=40, nonneg=False, warm=None):
    """argmin_x 0.5||x - b||^2 + lam TV(x)   (Beck & Teboulle 2009, FGP)."""
    if lam <= 0:
        return (np.maximum(b, 0) if nonneg else b), warm
    P = (lambda v: np.maximum(v, 0)) if nonneg else (lambda v: v)
    if warm is None:
        r = np.zeros_like(b); s = np.zeros_like(b)
    else:
        r, s = warm
    p_old, q_old, t_old = r, s, 1.0
    for _ in range(iters):
        sol = P(b - lam * _L(r, s))
        tr, ts = _Lt(sol)
        r = r + tr / (8 * lam)
        s = s + ts / (8 * lam)
        w = np.maximum(1.0, np.sqrt(r * r + s * s))
        p, q = r / w, s / w
        t = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2
        r = p + (t_old - 1) / t * (p - p_old)
        s = q + (t_old - 1) / t * (q - q_old)
        p_old, q_old, t_old = p, q, t
    return P(b - lam * _L(p_old, q_old)), (p_old, q_old)


def fista_tv(G, b, shape, lam, x0=None, iters=400, nonneg=False, tv_iters=30, L=None):
    """minimise 0.5 x'Gx - b'x + lam TV(x) with FISTA; G = s^2 A'A, b = s A'y."""
    if L is None:
        L = float(np.linalg.eigvalsh(G)[-1])
    x = np.zeros(G.shape[0]) if x0 is None else x0.ravel().copy()
    y, t, warm = x.copy(), 1.0, None
    for _ in range(iters):
        g = G @ y - b
        z, warm = prox_tv((y - g / L).reshape(shape), lam / L, tv_iters, nonneg, warm)
        xn = z.ravel()
        tn = (1 + np.sqrt(1 + 4 * t * t)) / 2
        y = xn + (t - 1) / tn * (xn - x)
        x, t = xn, tn
    return x.reshape(shape)


# ------------------------------------------------------------ one-call API --
class Reconstructor:
    """Precomputes A~'A~ once; reconstructs any number of photos / lambdas."""

    def __init__(self, A, n, shape, background="poly", order=1):
        self.n, self.shape, self.background, self.order = n, shape, background, order
        self.At = background_free(A, n, background, order)
        self.AtA = self.At.T @ self.At
        self.Lmax = float(np.linalg.eigvalsh(self.AtA)[-1])

    def prepare(self, Y):
        return background_free(photo_columns(Y), self.n, self.background, self.order)

    def solve(self, Y, scale=(1.0, 1.0, 1.0), lam=(0.0, 0.0, 0.0), iters=400, nonneg=True, init="ls"):
        ys = self.prepare(Y)
        out = []
        for c in range(ys.shape[1]):
            s = scale[c]
            G = s * s * self.AtA
            b = s * (self.At.T @ ys[:, c])
            x0 = None
            if init == "ls":
                x0 = np.linalg.lstsq(G, b, rcond=None)[0]
            out.append(fista_tv(G, b, self.shape, lam[c], x0, iters, nonneg, L=s * s * self.Lmax))
        return np.stack(out, -1)

    def residual(self, Y, X, scale=(1.0, 1.0, 1.0)):
        """|| s A~ x - y~ || per channel (what the model leaves unexplained)."""
        ys = self.prepare(Y)
        return np.array([np.linalg.norm(scale[c] * self.At @ X[:, :, c].ravel() - ys[:, c])
                         for c in range(ys.shape[1])])

    def discrepancy_lambda(self, Y, sigma, scale=(1.0, 1.0, 1.0), tau=1.0, grid=None, iters=250):
        """Largest lambda whose data misfit stays within tau * sigma * sqrt(N) (Morozov)."""
        N = self.n * self.n
        grid = np.logspace(-4, 2, 13) if grid is None else grid
        ys = self.prepare(Y)
        lams = []
        for c in range(ys.shape[1]):
            s = scale[c]
            G, b = s * s * self.AtA, s * (self.At.T @ ys[:, c])
            x0 = np.linalg.lstsq(G, b, rcond=None)[0]
            # lambda is relative to ||b||, so the grid is scale free
            base = np.abs(b).max()
            best = grid[0] * base
            target = tau * sigma[c] * np.sqrt(N) * noise_gain(self.background)
            for g in grid:
                lam = g * base
                x = fista_tv(G, b, self.shape, lam, x0, iters, True, L=s * s * self.Lmax)
                if np.linalg.norm(s * self.At @ x.ravel() - ys[:, c]) <= target:
                    best = lam
                else:
                    break
            lams.append(best)
        return lams
