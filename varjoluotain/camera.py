"""
A simple camera: what a photo of the visible wall looks like.

expected electrons = k * gain_c * (A x_c) + room light
photo              = Poisson(expected) + Normal(0, read_noise)

`photons` is the expected electron count at the brightest wall pixel for a
fully white hidden screen; it sets the signal-to-noise ratio.  The published
measurements average 20 raw frames and bin 8 x 8 raw pixels per colour, so
around 1e6 electrons per binned pixel is realistic; lower values mimic a
single dim frame.  Room light is a tilted plane: a constant plus a linear
ramp across the field of view.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Camera:
    photons: float = 1e6
    read_noise: float = 30.0
    ambient: float = 0.2          # room light, as a fraction of the white-screen peak
    ambient_tilt: float = 0.3     # relative change of room light across the view
    gain: tuple = (1.0, 1.0, 1.0) # relative channel sensitivity x screen colour balance
    seed: int = 0


def exposure(A: np.ndarray, cam: Camera) -> float:
    """Electrons per unit of A x for this camera (from the white-screen peak)."""
    return cam.photons / float((A @ np.ones(A.shape[1])).max())


def room_light(n: int, cam: Camera) -> np.ndarray:
    u = np.linspace(-1, 1, n)
    X, Z = np.meshgrid(u, -u)
    return cam.ambient * cam.photons * (1 + cam.ambient_tilt * (0.8 * X + 0.6 * Z))


def photograph(A: np.ndarray, scene: np.ndarray, n: int, cam: Camera, k: float = None):
    """scene: rows x cols x 3 in [0, 1]. Returns (photo n x n x 3, noise sigma per channel, k).
    k (electrons per unit of A x) defaults to this A's white-screen peak; pass a
    fixed k to compare rooms under the same camera exposure."""
    rng = np.random.default_rng(cam.seed)
    k = exposure(A, cam) if k is None else k
    amb = room_light(n, cam)
    photo, sig = [], []
    for c in range(scene.shape[2]):
        mean = k * cam.gain[c] * (A @ scene[:, :, c].ravel()).reshape(n, n) + amb
        photo.append(rng.poisson(np.maximum(mean, 0)) + rng.normal(0, cam.read_noise, mean.shape))
        sig.append(float(np.sqrt(mean.mean() + cam.read_noise ** 2)))
    return np.stack(photo, -1), np.array(sig), k


def scales(A: np.ndarray, cam: Camera, k: float = None):
    """Channel scales that turn A into electrons (what calibration would give)."""
    k = exposure(A, cam) if k is None else k
    return tuple(k * g for g in cam.gain)
