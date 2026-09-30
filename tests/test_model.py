"""Checks for the forward model, the solver and the funnel's building blocks.
Run:  python -m pytest -q tests"""
import os, sys
from dataclasses import replace
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from varjoluotain import geometry as G, transport as T, solve as S, camera as C, images as I, locate as L, ledger as Lg


def small(**kw):
    """A cheap version of the D11 room for tests."""
    return L.model_at(G.d11((0.475, 0.570, 0.214), **kw), 6, 8, 6, 2)


def test_vectorised_build_matches_per_patch():
    s = small()
    A = T.build_A(s)
    for (i, j) in [(0, 0), (2, 5), (5, 7)]:
        assert np.allclose(A[:, i * s.cols + j], T.patch_response(s, i, j).ravel(), rtol=1e-12, atol=0)


def test_occluder_only_removes_light():
    s = small()
    A, A0 = T.build_A(s), T.build_A(replace(s, occluders=[]))
    assert np.all(A <= A0 + 1e-15) and A.sum() < A0.sum()


def test_mask_equals_rectangle():
    s = small()
    plate = s.occluders[0]
    mask = G.MaskOccluder(plate.x0, plate.z0, plate.x1 - plate.x0, plate.z1 - plate.z0, plate.y,
                          np.ones((30, 30), bool))
    A_rect = T.build_A(replace(s, occluders=[plate]))
    A_mask = T.build_A(replace(s, occluders=[], masks=[mask]))
    # a full mask and the rectangle agree except where a shadow edge falls between samples
    assert np.mean(np.isclose(A_rect, A_mask, rtol=1e-9, atol=1e-18)) > 0.99


def test_background_operators_cancel_room_light():
    n = 21
    ramp = S.ramp_basis(n, 1) @ np.array([5.0, -2.0, 3.0])
    assert np.abs(S.background_free(ramp[:, None], n, "poly")).max() < 1e-9
    const = np.full((n * n, 1), 7.0)
    d = S.background_free(const, n, "diff")
    assert np.abs(d[:-1]).max() < 1e-12


def test_noise_free_reconstruction_is_close():
    s = L.model_at(G.d11((0.475, 0.570, 0.214)), 9, 12, 3, 3)
    A = T.build_A(s)
    x = I.sample("astronaut", 9, 12)
    Y = np.stack([(A @ x[:, :, c].ravel()).reshape(s.wall_n, s.wall_n) for c in range(3)], -1)
    rec = S.Reconstructor(A, s.wall_n, (9, 12))
    X = rec.solve(Y, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), iters=3000, nonneg=True)
    assert np.sqrt(np.mean((X - x) ** 2)) < 0.02


def test_occluder_makes_more_modes_observable():
    s = L.model_at(G.d11((0.475, 0.570, 0.214)), 9, 12, 3, 3)
    cam = C.Camera(photons=1e6)
    counts = []
    for room in (s, replace(s, occluders=[])):
        A = T.build_A(room)
        sv, _ = Lg.modes(S.background_free(A, room.wall_n))
        _, sig, k = C.photograph(A, I.sample("coffee", 9, 12), room.wall_n, cam)
        counts.append(int(Lg.observable(sv, k, sig.mean()).sum()))
    assert counts[0] > 3 * counts[1]


def test_cheap_models_cover_the_same_screen():
    s = G.d11()
    for (r, c) in [(6, 8), (12, 15)]:
        assert np.allclose(L.model_at(s, r, c, 3, 3).screen_extent(), s.screen_extent())


def test_uvy_round_trip():
    s = G.d11()
    corner = np.array([0.4733, 0.5661, 0.2072])
    assert np.allclose(L.corner_from_uvy(s, L.uvy_from_corner(s, corner)), corner)


def test_true_position_beats_neighbours():
    """The misfit the funnel uses is smallest at the true occluder (cheap model, noisy photo)."""
    truth = np.array((0.475, 0.570, 0.214))
    full = G.d11(tuple(truth))
    m = L.model_at(full, 6, 8, 3, 3)
    A = T.build_A(L.model_at(full, 12, 15, 3, 3))
    x = I.sample("rocket", 12, 15)
    Y = np.stack([(A @ x[:, :, c].ravel()).reshape(42, 42) for c in range(3)], -1)
    Y = Y + np.random.default_rng(0).normal(0, 1e-3 * Y.std(), Y.shape)
    cost = lambda h: L.misfit(T.build_A(replace(m, occluders=G.square_on_stand(*h))), Y, m.wall_n)
    c0 = cost(truth)
    for d in ([0.02, 0, 0], [0, 0, 0.02], [-0.02, 0, 0]):
        assert cost(truth + np.array(d)) > c0


@pytest.mark.skipif(not os.environ.get("ORDINARY_CAMERA"), reason="set ORDINARY_CAMERA to a clone of the paper's repository")
def test_real_bu_photo():
    from varjoluotain import realdata as R
    root = R.find_d11(os.environ["ORDINARY_CAMERA"])
    fname, occ, scale, lam = R.D11_SCENES["bu"]
    photo, gt, _ = R.load_measurement(os.path.join(root, fname))
    A = T.build_A(G.paper_compatible(G.d11(occ)))
    X = S.Reconstructor(A, 126, (29, 36), background="diff").solve(photo, scale, lam, 300, nonneg=False)
    a, b = X.ravel() - X.mean(), gt.astype(float).ravel() - gt.mean()
    assert a @ b / np.linalg.norm(a) / np.linalg.norm(b) > 0.5
