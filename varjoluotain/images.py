"""Hidden scenes: ordinary digital images shown on the hidden screen."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SAMPLES = ("astronaut", "coffee", "chelsea", "rocket", "colorwheel", "hubble_deep_field")


def fit(img: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Centre-crop to the screen's aspect ratio and area-average to rows x cols, in [0, 1]."""
    if img.ndim == 2:
        img = np.stack([img] * 3, -1)
    img = img[:, :, :3]
    h, w = img.shape[:2]
    target = cols / rows
    if w / h > target:
        nw = int(round(h * target)); x0 = (w - nw) // 2
        img = img[:, x0:x0 + nw]
    else:
        nh = int(round(w / target)); y0 = (h - nh) // 2
        img = img[y0:y0 + nh]
    pil = Image.fromarray(img.astype(np.uint8))
    return np.asarray(pil.resize((cols, rows), Image.BOX), dtype=np.float64) / 255.0


def sample(name: str, rows: int = 29, cols: int = 36) -> np.ndarray:
    from skimage import data
    return fit(getattr(data, name)(), rows, cols)


def load(path: str, rows: int = 29, cols: int = 36) -> np.ndarray:
    return fit(np.asarray(Image.open(path).convert("RGB")), rows, cols)


def text(word: str = "LUOTAIN", rows: int = 29, cols: int = 36, color=(1.0, 0.85, 0.2)) -> np.ndarray:
    W, H = cols * 8, rows * 8
    big = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(big)
    size = H // 2
    while True:                                   # largest font that fits with a margin
        try:
            font = ImageFont.load_default(size=size)
        except TypeError:
            font = ImageFont.load_default(); break
        box = d.textbbox((0, 0), word, font=font)
        if box[2] - box[0] <= 0.88 * W or size <= 8:
            break
        size -= 2
    box = d.textbbox((0, 0), word, font=font)
    d.text(((W - (box[2] - box[0])) / 2 - box[0], (H - (box[3] - box[1])) / 2 - box[1]), word, fill=255, font=font)
    m = np.asarray(big.resize((cols, rows), Image.BOX), dtype=np.float64) / 255.0
    return m[:, :, None] * np.array(color)[None, None, :]
