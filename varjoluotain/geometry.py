"""
Scene geometry for computational periscopy.

Coordinates (metres):  x runs along the wall, y is depth, z is up.
The hidden scene is a planar light source (an LCD monitor) in the plane y = 0,
facing the visible wall, which is the plane y = D.  An opaque occluder sits
somewhere in between.  The camera cannot see the monitor; it photographs a
square patch of the wall (its field of view).

Default numbers are the "D11" configuration of Saunders, Murray-Bruce & Goyal,
Nature 565, 472 (2019): monitor 1.03 m from the wall, a 7.5 cm square occluder
on a thin stand, a 43.7 cm camera field of view, a 29 x 36 grid of scene
patches of 35 x 35 monitor pixels each.
"""
from dataclasses import dataclass, field, replace
from typing import List, Optional
import numpy as np


@dataclass
class Rect:
    """Opaque axis-aligned rectangle in a plane parallel to the wall, at depth y."""
    x0: float
    x1: float
    z0: float
    z1: float
    y: float


@dataclass
class MaskOccluder:
    """Opaque planar mask at depth y. mask[i, j] True = opaque; row 0 = top (max z),
    column 0 = min x. Covers [x0, x0 + w] x [z0, z0 + h]."""
    x0: float
    z0: float
    w: float
    h: float
    y: float
    mask: np.ndarray


@dataclass
class Setup:
    D: float = 1.03                     # monitor-to-wall distance
    # camera field of view on the wall
    wall_x0: float = 0.521
    wall_z0: float = 0.048
    wall_w: float = 0.4372
    wall_h: float = 0.4372
    wall_n: int = 126                   # wall samples per side (binned camera pixels)
    # hidden scene: grid of monitor patches
    rows: int = 29
    cols: int = 36
    block_w: float = 0.408 / 1280 * 35
    block_h: float = 0.3085 / 1024 * 35
    mon_x0: float = 0.0210 + 0.408 / 1280 * (1280 % 36)
    mon_z0: float = 0.136 + 0.3085 / 1024 * (1024 % 29)
    sub: int = 6                        # point sources per patch side (penumbra accuracy)
    anchor_mode: str = "paper"          # "paper": published patch spacing (small gaps); "tile": patches tile exactly
    # LCD brightness falls off with viewing angle: cos(angle)^power
    lcd_power_x: float = 1.0
    lcd_power_z: float = 18.0
    lcd_mirror_quirk: bool = False      # True reproduces the published code's model exactly (see transport.py)
    visibility_convention: str = "center"   # "paper" reproduces the published pixel convention
    occluders: List[Rect] = field(default_factory=list)
    masks: List[MaskOccluder] = field(default_factory=list)

    # ------------------------------------------------------------------
    def wall_x(self):
        return np.linspace(self.wall_x0, self.wall_x0 + self.wall_w, self.wall_n)

    def wall_z(self):
        return np.linspace(self.wall_z0, self.wall_z0 + self.wall_h, self.wall_n)

    def block_anchor_x(self):
        """x of each patch's right edge, by monitor column (ascending x); the patch spans [a - block_w, a]."""
        if self.anchor_mode == "tile":
            return self.mon_x0 + np.arange(self.cols) * self.block_w
        return np.linspace(self.mon_x0, self.mon_x0 + self.cols * self.block_w, self.cols)

    def block_anchor_z(self):
        """z of each patch's top edge, by row (row 0 = top); the patch spans [a - block_h, a]."""
        if self.anchor_mode == "tile":
            return self.mon_z0 + (self.rows - 1 - np.arange(self.rows)) * self.block_h
        return np.linspace(self.mon_z0 + self.rows * self.block_h, self.mon_z0, self.rows)

    def screen_extent(self):
        """(x_start, x_end, z_start, z_end) of the lit screen area."""
        ax, az = self.block_anchor_x(), self.block_anchor_z()
        return ax[0] - self.block_w, ax[-1], az[-1] - self.block_h, az[0]

    def with_occluder(self, *rects: Rect, masks=()):
        return replace(self, occluders=list(rects), masks=list(masks))


def square_on_stand(llx: float, lly: float, llz: float, size: float = 0.075,
                    stand_w: float = 0.0055) -> List[Rect]:
    """The paper's occluder: a square plate with its lower-left corner at
    (llx, lly, llz), held up by a thin vertical stand from the floor."""
    plate = Rect(llx, llx + size, llz, llz + size, lly)
    cx = llx + size / 2
    stand = Rect(cx - stand_w / 2, cx + stand_w / 2, 0.0, llz, lly)
    return [plate, stand]


def d11(occluder_llcorner=(0.4733, 0.5661, 0.2072), **kw) -> Setup:
    """The paper's D11 experiment. The default occluder corner is the paper's
    own estimate for its 'BU' measurement (measured tape position: 0.475, 0.570, 0.214)."""
    s = Setup(**kw)
    return replace(s, occluders=square_on_stand(*occluder_llcorner))


def paper_compatible(setup: Setup) -> Setup:
    """Switch on the two conventions of the published MATLAB model, needed to
    reproduce its real-photo reconstructions with its calibrated constants."""
    return replace(setup, lcd_mirror_quirk=True, visibility_convention="paper")
