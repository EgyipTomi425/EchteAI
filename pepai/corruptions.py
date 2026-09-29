"""Traffic-relevant image corruptions (COCO-C protocol, Michaelis et al. 2019) plus low light."""
import numpy as np

# imagecorruptions still references np.float_, which NumPy 2 removed.
if not hasattr(np, "float_"):
    np.float_ = np.float64

from imagecorruptions import corrupt  # noqa: E402

# name -> short description used in tables/figures
CONDITIONS = {
    "fog": "fog",
    "snow": "snow",
    "frost": "frost (windshield)",
    "dark": "low light",
    "brightness": "glare / overexposure",
    "contrast": "low contrast (haze)",
    "gaussian_noise": "sensor noise",
    "motion_blur": "motion blur",
}

SEVERITIES = (1, 2, 3, 4, 5)


def darken(img, severity):
    """Low-light imaging: gamma darkening plus photon (shot) noise, both growing with severity."""
    gamma = (1.5, 2.0, 2.5, 3.0, 3.5)[severity - 1]
    photons = (60, 30, 15, 8, 4)[severity - 1]
    x = (img.astype(np.float64) / 255.0) ** gamma
    rng = np.random.default_rng(int(img.sum()) % (2 ** 32))
    x = rng.poisson(x * photons) / photons
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def apply(img, condition, severity):
    """img: HxWx3 uint8 RGB array."""
    if condition == "dark":
        return darken(img, severity)
    out = corrupt(img, corruption_name=condition, severity=severity)
    return out[: img.shape[0], : img.shape[1]].astype(np.uint8)


def corrupt_file(job):
    """Worker: (src path, dst path, condition, severity, seed) -> writes the corrupted PNG."""
    from PIL import Image
    src, dst, condition, severity, seed = job
    if dst.exists():
        return
    np.random.seed(seed)          # imagecorruptions draws from the global NumPy RNG
    img = np.asarray(Image.open(src).convert("RGB"))
    Image.fromarray(apply(img, condition, severity)).save(dst)
