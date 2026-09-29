"""Download the calibration subset of COCO train2017 and ImageNetV2 (matched-frequency).

COCO val2017 and the annotations are expected under data/coco (see README).
"""
import json
import random
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from pepai.config import load_config

IMAGENETV2_URL = (
    "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/"
    "imagenetv2-matched-frequency.tar.gz"
)


def fetch(url, dst):
    if dst.exists():
        return
    tmp = dst.with_suffix(dst.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dst)


def coco_calibration_subset(cfg):
    root = cfg["coco"]["root"]
    with open(root / "annotations" / "instances_train2017.json") as f:
        images = json.load(f)["images"]
    images = sorted(images, key=lambda im: im["id"])
    rng = random.Random(cfg["seed"])
    chosen = rng.sample(images, cfg["coco"]["n_calib"])

    out_dir = root / "calib_train2017"
    out_dir.mkdir(exist_ok=True)
    with open(root / "calib_train2017.json", "w") as f:
        json.dump([im["file_name"] for im in chosen], f, indent=0)

    jobs = [(im["coco_url"], out_dir / im["file_name"]) for im in chosen]
    with ThreadPoolExecutor(16) as pool:
        list(tqdm(pool.map(lambda j: fetch(*j), jobs), total=len(jobs), desc="COCO calib"))


def imagenetv2(cfg):
    root = cfg["imagenetv2"]["root"]
    root.mkdir(parents=True, exist_ok=True)
    if (root / "imagenetv2-matched-frequency-format-val").exists():
        return
    archive = root / "imagenetv2-matched-frequency.tar.gz"
    print("Downloading ImageNetV2 (1.2 GB)...")
    fetch(IMAGENETV2_URL, archive)
    with tarfile.open(archive) as tar:
        tar.extractall(root, filter="data")
    archive.unlink()


if __name__ == "__main__":
    cfg = load_config()
    coco_calibration_subset(cfg)
    imagenetv2(cfg)
