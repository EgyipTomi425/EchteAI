"""Downstream effect of quantization-induced box shifts on monocular distance estimation (CPU only).

A flat-ground pinhole camera at height h_cam estimates the distance of an object from the image row
of its box bottom: Z = f * h_cam / (v_b - v_0). A bottom-edge shift dv changes the estimate by
dZ ~= Z^2 / (f * h_cam) * dv. Expressing the shift relative to the box height, dv = r * f * H / Z for
an object of physical height H, gives a relative distance error that depends neither on the focal
length nor on the distance:

    |dZ| / Z = |r| * H / h_cam.

The relative shifts r are measured between FP32 and the deployed engines (same-class greedy match at
IoU >= 0.5, FP32 score >= 0.5) on all 5 000 COCO val2017 images; H (pedestrian 1.7 m, car 1.5 m) and
h_cam (1.65 m, KITTI) are illustrative assumptions stated in the manuscript.
"""
import json
from collections import defaultdict

import numpy as np
import pandas as pd

from pepai.agreement import greedy_match
from pepai.config import load_config, results_dir

CAMERA_HEIGHT_M = 1.65
CLASSES = {"person": (1, 1.7), "car": (3, 1.5)}          # COCO category id, physical height (m)
DISTANCES_M = [20, 40]
DETECTORS = ["frcnn_r50_fpn", "yolov10s", "yolov10x"]
PRECISIONS = ["fp16", "int8", "fp8"]
SCORE_THR, IOU_THR = 0.5, 0.5


def by_image(path):
    out = defaultdict(list)
    for d in json.loads(path.read_text()):
        if d["score"] >= SCORE_THR:
            out[d["image_id"]].append(d)
    return out


def arrays(dets):
    if not dets:
        return np.zeros((0, 4)), np.zeros(0, int), np.zeros(0)
    b = np.array([d["bbox"] for d in dets], float)
    b[:, 2:] += b[:, :2]                                   # xywh -> xyxy
    return b, np.array([d["category_id"] for d in dets]), np.array([d["score"] for d in dets])


def relative_bottom_shifts(ref, quant):
    """Per matched FP32 detection: class and bottom-edge shift divided by the FP32 box height."""
    cls, rel, n_ref = [], [], 0
    for img, dets in ref.items():
        fb, fl, fs = arrays(dets)
        qb, ql, _ = arrays(quant.get(img, []))
        n_ref += len(fb)
        iou, match = greedy_match(fb, fl, fs, qb, ql, IOU_THR)
        ok = iou >= IOU_THR
        h = fb[ok, 3] - fb[ok, 1]
        rel.extend(((qb[match[ok], 3] - fb[ok, 3]) / np.maximum(h, 1e-6)).tolist())
        cls.extend(fl[ok].tolist())
    return np.array(cls), np.array(rel), n_ref


if __name__ == "__main__":
    cfg = load_config()
    det_dir = results_dir(cfg, "detections")
    rows = []
    for name in DETECTORS:
        f32 = det_dir / f"{name}_fp32.json"
        if not f32.exists():
            continue
        ref = by_image(f32)
        for precision in PRECISIONS:
            path = det_dir / f"{name}_{precision}.json"
            if not path.exists():
                continue
            cls, rel, n_ref = relative_bottom_shifts(ref, by_image(path))
            for group, (cat, height) in CLASSES.items():
                r = np.abs(rel[cls == cat])
                if len(r) == 0:
                    continue
                dz = 100 * r * height / CAMERA_HEIGHT_M       # relative distance error (%)
                row = {"model": name, "precision": precision, "class": group, "n_matched": len(r),
                       "match_rate_all": len(rel) / max(n_ref, 1),
                       "shift_rel_median_pct": 100 * np.median(r), "shift_rel_p95_pct": 100 * np.percentile(r, 95),
                       "dz_rel_median_pct": np.median(dz), "dz_rel_p95_pct": np.percentile(dz, 95),
                       "dz_rel_p99_pct": np.percentile(dz, 99), "share_dz_gt_5pct": (dz > 5).mean()}
                for z in DISTANCES_M:
                    row[f"dz_cm_p95_at_{z}m"] = np.percentile(dz, 95) * z        # % of Z metres -> cm
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(results_dir(cfg, "tables") / "localization.csv", index=False)
    print(df.round(3).to_string(index=False))
