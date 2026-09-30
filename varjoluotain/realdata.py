"""
Loader for the real camera measurements published with
Saunders, Murray-Bruce & Goyal, "Computational periscopy with an ordinary
digital camera", Nature 565, 472-475 (2019):
https://github.com/Computational-Periscopy/Ordinary-Camera

That repository carries no licence, so nothing of it is copied here: point
the scripts at your own clone.  The constants below (occluder estimates,
channel scales, TV weights) are the published values for the D11 scenes.
"""
import os
import numpy as np
import scipy.io as sio

# scene key -> (file, occluder lower-left corner estimate, channel scales, TV weights)
_S = 1.0 / 36.0                      # published scales are per 6 x 6 point sources
D11_SCENES = {
    "mushroom": ("image_test_mushroom20.mat", (0.4583, 0.5408, 0.2026),
                 (1.4063e4 * _S * 0.9, 1.5313e4 * _S * 0.98, 16250 * _S * 1.04),
                 (0.51e8, 0.561e8, 2.448e8)),
    "tommy":    ("image_test_smilehat20.mat", (0.4569, 0.5744, 0.2080),
                 (1.1406e4 * _S * 0.9, 1.3594e4 * _S * 0.98, 1.9063e4 * _S * 1.04),
                 (5.72e7, 6.76e7, 5.72e7)),
    "bu":       ("image_test_bur20.mat", (0.4733, 0.5661, 0.2072),
                 (12500 * _S * 0.9, 15625 * _S * 0.98, 1.7188e4 * _S * 1.04),
                 (5e7, 25e7, 25e7)),
    "rgb":      ("image_test_colbar20.mat", (0.4693, 0.5629, 0.2080),
                 (1.4063e4 * _S * 0.9, 1.4063e4 * _S * 0.98, 1.4063e4 * _S * 1.04),
                 (52.5e6, 50e6, 47.5e6)),
}
D11_TAPE_MEASURED = (0.475, 1.03 - 0.460, 0.214)   # the occluder position measured by hand


def _bayer_rgb(raw):
    r = raw[0::2, 0::2]
    g = (raw[1::2, 0::2] + raw[0::2, 1::2]) / 2
    b = raw[1::2, 1::2]
    return np.stack([r, g, b], -1)


def _bin2(im, levels):
    for _ in range(levels):
        im = (im[0::2, 0::2] + im[0::2, 1::2] + im[1::2, 0::2] + im[1::2, 1::2]) / 4
    return im


def load_measurement(path, levels=3):
    """Returns (photo n x n x 3, ground truth 29 x 36 x 3 uint8, raw metadata).
    Row 0 of the photo is the top of the wall, column 0 the smallest x."""
    m = sio.loadmat(path)
    raw = m["image"].astype(np.float64)
    photo = _bin2(_bayer_rgb(raw), levels)
    meta = {k: np.asarray(m[k]).ravel()[0] for k in ("shutter_speed", "iterations") if k in m}
    return photo, m["ground_truth"], meta


def find_d11(root):
    for cand in (root, os.path.join(root, "Data", "TestPosD11"), os.path.join(root, "TestPosD11")):
        if os.path.exists(os.path.join(cand, "image_test_bur20.mat")):
            return cand
    raise FileNotFoundError("TestPosD11 data not found under %s" % root)


def raw_noise_sigma(path, levels=3):
    """Rough per-binned-pixel noise level: robust spread of the finest-scale
    differences of the raw frame (the wall is smooth at the raw pixel scale),
    divided by the number of raw values averaged into one binned pixel."""
    raw = sio.loadmat(path)["image"].astype(np.float64)
    rgb = _bayer_rgb(raw)
    out = []
    for c in range(3):
        ch = rgb[:, :, c]
        d = (ch[:-1:2, :-1:2] - ch[1::2, :-1:2] - ch[:-1:2, 1::2] + ch[1::2, 1::2]) / 2   # ~ sigma
        mad = np.median(np.abs(d - np.median(d))) * 1.4826
        out.append(mad / np.sqrt(4 ** levels))          # 2x2 binning, `levels` times
    return np.array(out)
