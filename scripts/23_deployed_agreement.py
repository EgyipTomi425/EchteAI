"""FP32 agreement of the deployed engines (FP16, INT8, FP8) on all evaluation images (CPU only).

Detectors: image-level strict and score-tolerant disagreement (pepai.agreement.image_disagreement) and
the number of FP32 detections above the operating threshold that vanish, i.e. have no same-class
counterpart with IoU >= 0.5 even down to a score of 0.3, in total and for road users (person, bicycle,
car, motorcycle, bus, truck). Classifiers: share of images whose top-1 class differs from FP32.
These counts are the critical-error term of the AIBO cost model (11_aibo.py).
"""
import json
from collections import defaultdict

import numpy as np
import pandas as pd

from pepai import data
from pepai.agreement import greedy_match, image_disagreement
from pepai.config import load_config, results_dir

DETECTORS = ["frcnn_r50_fpn", "yolov10s", "yolov10x"]
CLASSIFIERS = ["efficientnet_b0", "densenet121"]
PRECISIONS = ["fp16", "int8", "fp8"]
ROAD_USERS = {1, 2, 3, 4, 6, 8}           # COCO ids: person, bicycle, car, motorcycle, bus, truck
SCORE_THR, TOLERANCE, IOU_THR = 0.5, 0.2, 0.5


def load(path):
    out = defaultdict(lambda: {"boxes": [], "labels": [], "scores": []})
    for d in json.loads(path.read_text()):
        x, y, w, h = d["bbox"]
        o = out[d["image_id"]]
        o["boxes"].append([x, y, x + w, y + h])
        o["labels"].append(d["category_id"])
        o["scores"].append(d["score"])
    return {k: {"boxes": np.array(v["boxes"], float).reshape(-1, 4), "labels": np.array(v["labels"], int),
                "scores": np.array(v["scores"], float)} for k, v in out.items()}


EMPTY = {"boxes": np.zeros((0, 4)), "labels": np.zeros(0, int), "scores": np.zeros(0)}


def ground_truth(coco, img_id):
    anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=False))
    boxes = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                      for a in anns]).reshape(-1, 4)
    return boxes, np.array([a["category_id"] for a in anns], int)


def vanished(ref, quant):
    """Per FP32 detection above the threshold: (class, vanished under the tolerant definition)."""
    keep = ref["scores"] >= SCORE_THR
    fb, fl, fs = ref["boxes"][keep], ref["labels"][keep], ref["scores"][keep]
    qk = quant["scores"] >= SCORE_THR - TOLERANCE
    iou, _ = greedy_match(fb, fl, fs, quant["boxes"][qk], quant["labels"][qk], IOU_THR)
    return fl, iou < IOU_THR


if __name__ == "__main__":
    cfg = load_config()
    det_dir = results_dir(cfg, "detections")
    coco, ids = data.coco_val_ids(cfg)
    rows = []
    for name in DETECTORS:
        f32 = det_dir / f"{name}_fp32.json"
        if not f32.exists():
            continue
        ref = load(f32)
        for precision in PRECISIONS:
            path = det_dir / f"{name}_{precision}.json"
            if not path.exists():
                continue
            q = load(path)
            err = err_tol = n_det = n_van = n_road = n_road_van = 0
            for img in ids:
                r, qq = ref.get(img, EMPTY), q.get(img, EMPTY)
                gt_boxes, gt_labels = ground_truth(coco, img)
                d = image_disagreement(r, qq, gt_boxes, gt_labels, SCORE_THR, IOU_THR, TOLERANCE)
                err += d["error"]
                err_tol += d["error_tol"]
                labels, van = vanished(r, qq)
                road = np.isin(labels, list(ROAD_USERS))
                n_det += len(labels)
                n_van += int(van.sum())
                n_road += int(road.sum())
                n_road_van += int((van & road).sum())
            n = len(ids)
            rows.append({"model": name, "precision": precision, "n_images": n,
                         "image_error_rate": err / n, "image_error_rate_tol": err_tol / n,
                         "fp32_detections": n_det, "vanished": n_van, "vanished_share": n_van / max(n_det, 1),
                         "vanished_per_image": n_van / n, "road_user_detections": n_road,
                         "road_user_vanished": n_road_van, "road_user_vanished_per_image": n_road_van / n})
            print(rows[-1], flush=True)
    for name in CLASSIFIERS:
        f32 = det_dir / f"{name}_fp32_cls.npz"
        if not f32.exists():
            continue
        ref = np.load(f32)["preds"]
        for precision in PRECISIONS:
            path = det_dir / f"{name}_{precision}_cls.npz"
            if path.exists():
                flips = (np.load(path)["preds"] != ref).mean()
                rows.append({"model": name, "precision": precision, "n_images": len(ref),
                             "image_error_rate": flips, "image_error_rate_tol": flips})
                print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(results_dir(cfg, "tables") / "deployed_agreement.csv", index=False)
