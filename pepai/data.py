"""Dataset access: COCO (calibration subset of train2017, val2017) and ImageNetV2."""
import json
import random
from pathlib import Path

from PIL import Image


def coco_calib_paths(cfg):
    root = cfg["coco"]["root"]
    names = json.loads((root / "calib_train2017.json").read_text())
    return [root / "calib_train2017" / n for n in names]


def coco_val_ids(cfg, n=None):
    """val2017 image ids in a fixed, seeded order; the first n are used for activation analysis."""
    from pycocotools.coco import COCO
    coco = COCO(str(cfg["coco"]["root"] / "annotations" / "instances_val2017.json"))
    ids = sorted(coco.getImgIds())
    random.Random(cfg["seed"]).shuffle(ids)
    return coco, (ids[:n] if n else ids)


def coco_val_path(cfg, coco, img_id):
    return cfg["coco"]["root"] / "val2017" / coco.loadImgs(img_id)[0]["file_name"]


def imagenetv2_split(cfg):
    """(calibration, evaluation) lists of (path, label); disjoint, seeded."""
    root = Path(cfg["imagenetv2"]["root"]) / "imagenetv2-matched-frequency-format-val"
    items = sorted((p, int(p.parent.name)) for p in root.glob("*/*.jpeg"))
    random.Random(cfg["seed"]).shuffle(items)
    n = cfg["imagenetv2"]["n_calib"]
    return items[:n], items[n:]


def load_rgb(path):
    return Image.open(path).convert("RGB")
