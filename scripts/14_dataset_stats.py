"""Dataset table: images, objects and category coverage of every data split used (CPU only)."""
import json
from collections import Counter

import numpy as np
import pandas as pd
from pycocotools.coco import COCO

from pepai import data
from pepai.config import load_config, results_dir

TRAFFIC = ["person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light", "stop sign"]


def coco_split_stats(coco, ids, label):
    anns = [a for a in coco.loadAnns(coco.getAnnIds(imgIds=ids, iscrowd=False))]
    cats = Counter(coco.loadCats(a["category_id"])[0]["name"] for a in anns)
    areas = np.array([a["area"] for a in anns])
    per_img = Counter(a["image_id"] for a in anns)
    row = {"split": label, "images": len(ids), "objects": len(anns),
           "objects_per_image": len(anns) / len(ids),
           "categories_present": len(cats),
           "small_pct": 100 * (areas < 32 ** 2).mean(), "medium_pct": 100 * ((areas >= 32 ** 2) & (areas < 96 ** 2)).mean(),
           "large_pct": 100 * (areas >= 96 ** 2).mean(),
           "images_without_objects": sum(1 for i in ids if per_img[i] == 0),
           "traffic_objects_pct": 100 * sum(cats[c] for c in TRAFFIC) / max(len(anns), 1)}
    for c in TRAFFIC:
        row[f"n_{c.replace(' ', '_')}"] = cats[c]
    return row


if __name__ == "__main__":
    cfg = load_config()
    root = cfg["coco"]["root"]
    train = COCO(str(root / "annotations" / "instances_train2017.json"))
    names = json.loads((root / "calib_train2017.json").read_text())
    by_name = {im["file_name"]: im["id"] for im in train.dataset["images"]}
    calib_ids = [by_name[n] for n in names]

    val, val_ids = data.coco_val_ids(cfg)
    act_ids = val_ids[:cfg["coco"]["n_activation"]]
    rows = [coco_split_stats(train, calib_ids, "COCO train2017 calibration subset"),
            coco_split_stats(val, val_ids, "COCO val2017 (accuracy, deviation-risk)"),
            coco_split_stats(val, act_ids, "COCO val2017 analysis subset (activations, robustness)")]
    calib, evaluation = data.imagenetv2_split(cfg)
    for label, items in (("ImageNetV2 calibration", calib), ("ImageNetV2 evaluation", evaluation)):
        rows.append({"split": label, "images": len(items), "objects": np.nan,
                     "categories_present": len({lab for _, lab in items})})
    df = pd.DataFrame(rows)
    df.to_csv(results_dir(cfg, "tables") / "datasets.csv", index=False)
    print(df.round(1).to_string(index=False))
