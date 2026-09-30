"""
Light transport: how much light each hidden patch sends to each wall pixel.

For a point source on the monitor (plane y = 0) and a wall point (plane y = D)
at distance r, a Lambertian wall receives  cos(theta_src) cos(theta_wall) / r^2
= D^2 / r^4  (both planes are parallel).  An LCD also dims with viewing
angle, modelled as cos(angle)^p.  The occluder blocks the straight path;
for a plate parallel to the wall its shadow is the plate scaled by D / y_occ
about the source.  Each patch is split into sub x sub point sources, which
turns hard shadows into the penumbra the whole method relies on.

A has one row per wall pixel (C order of the wall image, row 0 = top of the
wall, column 0 = smallest x) and one column per scene patch (C order of the
scene image as the monitor displays it: row 0 = top, column 0 = viewer's left,
which is the monitor's largest-x column because the viewer faces -y).

Two switches reproduce the published MATLAB model bit-for-bit in convention:
  lcd_mirror_quirk      the published code applies the LCD vertical fall-off
                        with the wall's z axis reversed relative to the
                        geometry term; its calibrated constants assume this.
  visibility_convention "paper" tests shadow membership at poly2mask pixel
                        centres (x0 + (k+1) W/n) instead of the sample points.
"""
import numpy as np
from .geometry import Setup


def _rows_z(s: Setup):
    return s.wall_z()[::-1]                      # image row 0 = top of the wall


def _vis_coords(s: Setup):
    n = s.wall_n
    if s.visibility_convention == "paper":
        vx = s.wall_x0 + (np.arange(n) + 1) * s.wall_w / n
        vz = s.wall_z0 + (n - np.arange(n)) * s.wall_h / n
        return vx, vz
    return s.wall_x(), _rows_z(s)


def patch_sources(s: Setup, i: int, j: int):
    """Point sources of scene patch (display row i, display column j)."""
    mc = s.cols - 1 - j                          # monitor column (ascending x)
    ax, az = s.block_anchor_x()[mc], s.block_anchor_z()[i]
    k = np.arange(s.sub)
    lx = ax - k * s.block_w / s.sub
    lz = az - k * s.block_h / s.sub
    LX, LZ = np.meshgrid(lx, lz)
    return LX.ravel(), LZ.ravel(), (ax - s.block_w / 2, az - s.block_h / 2)


def visibility(s: Setup, lx, lz):
    """(S, n, n) visibility of every wall pixel from S point sources."""
    n = s.wall_n
    vx, vz = _vis_coords(s)
    V = np.ones((len(lx), n, n), dtype=bool)
    for r in s.occluders:
        k = s.D / r.y
        X1, X2 = lx + (r.x0 - lx) * k, lx + (r.x1 - lx) * k
        Z1, Z2 = lz + (r.z0 - lz) * k, lz + (r.z1 - lz) * k
        inx = (vx[None, :] >= np.minimum(X1, X2)[:, None]) & (vx[None, :] <= np.maximum(X1, X2)[:, None])
        inz = (vz[None, :] >= np.minimum(Z1, Z2)[:, None]) & (vz[None, :] <= np.maximum(Z1, Z2)[:, None])
        V &= ~(inz[:, :, None] & inx[:, None, :])
    for m in s.masks:
        f = m.y / s.D
        mh, mw = m.mask.shape
        ox = lx[:, None] + (vx[None, :] - lx[:, None]) * f           # where the ray crosses the mask plane
        oz = lz[:, None] + (vz[None, :] - lz[:, None]) * f
        ix = np.floor((ox - m.x0) / m.w * mw).astype(int)
        iz = np.floor((m.z0 + m.h - oz) / m.h * mh).astype(int)
        okx, okz = (ix >= 0) & (ix < mw), (iz >= 0) & (iz < mh)
        ix, iz = np.clip(ix, 0, mw - 1), np.clip(iz, 0, mh - 1)
        blocked = m.mask[iz[:, :, None], ix[:, None, :]] & okz[:, :, None] & okx[:, None, :]
        V &= ~blocked
    return V


def patch_response(s: Setup, i: int, j: int) -> np.ndarray:
    """Wall image (n x n) produced by scene patch (i, j) at unit brightness."""
    lx, lz, (cx, cz) = patch_sources(s, i, j)
    wx, zr = s.wall_x(), _rows_z(s)
    dx2 = (wx[None, :] - lx[:, None]) ** 2
    dz2 = (zr[None, :] - lz[:, None]) ** 2
    r2 = dz2[:, :, None] + dx2[:, None, :] + s.D ** 2
    inten = (s.D ** 2) / (r2 * r2)
    if s.occluders or s.masks:
        inten = inten * visibility(s, lx, lz)
    img = inten.sum(0)
    # LCD viewing-angle fall-off, evaluated once per patch as in the paper
    mx = np.cos(np.arctan((cx - wx) / s.D)) ** s.lcd_power_x
    zz = s.wall_z() if s.lcd_mirror_quirk else zr
    mz = np.cos(np.arctan((cz - zz) / s.D)) ** s.lcd_power_z
    return img * mz[:, None] * mx[None, :]


def build_A(s: Setup, dtype=np.float64, chunk_sources: int = 144) -> np.ndarray:
    """Full transport matrix, (wall_n^2) x (rows * cols).  Vectorised over
    patches in chunks; identical to stacking patch_response for every patch."""
    n, P, S = s.wall_n, s.rows * s.cols, s.sub * s.sub
    A = np.empty((n * n, P), dtype=dtype)
    wx, zr = s.wall_x(), _rows_z(s)
    zz = s.wall_z() if s.lcd_mirror_quirk else zr
    per = max(1, chunk_sources // S)
    patches = [(i, j) for i in range(s.rows) for j in range(s.cols)]
    for c0 in range(0, P, per):
        chunk = patches[c0:c0 + per]
        LX, LZ, CX, CZ = [], [], [], []
        for (i, j) in chunk:
            lx, lz, (cx, cz) = patch_sources(s, i, j)
            LX.append(lx); LZ.append(lz); CX.append(cx); CZ.append(cz)
        lx, lz = np.concatenate(LX), np.concatenate(LZ)
        dx2 = (wx[None, :] - lx[:, None]) ** 2
        dz2 = (zr[None, :] - lz[:, None]) ** 2
        r2 = dz2[:, :, None] + dx2[:, None, :] + s.D ** 2
        inten = (s.D ** 2) / (r2 * r2)
        if s.occluders or s.masks:
            inten *= visibility(s, lx, lz)
        img = inten.reshape(len(chunk), S, n, n).sum(1)
        mx = np.cos(np.arctan((np.array(CX)[:, None] - wx[None, :]) / s.D)) ** s.lcd_power_x
        mz = np.cos(np.arctan((np.array(CZ)[:, None] - zz[None, :]) / s.D)) ** s.lcd_power_z
        img *= mz[:, :, None] * mx[:, None, :]
        A[:, c0:c0 + len(chunk)] = img.reshape(len(chunk), n * n).T
    return A
