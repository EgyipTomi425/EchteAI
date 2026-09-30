"""Practical detection quality of the detectors per precision (CPU only, from stored detections).

COCO mAP@[.5:.95] averages over 80 classes and ten IoU thresholds up to 0.95, which understates how usable a
detector is for the road users that matter in traffic. From the stored detections of every engine this script
reports, for the road-user classes (person, bicycle, car, motorcycle, bus, truck):
  * AP@[.5:.95] and AP50 per class (COCOeval precision array),
  * recall and precision at the operating point (score >= 0.5, IoU >= 0.5), for all objects and for objects
    of at least 32x32 pixels (COCO medium and large), using the greedy score-ordered matching of COCOeval.
"""
import contextlib
import io
import json

import numpy as np
import pandas as pd
from pycocotools.cocoeval import COCOeval

from pepai import data
from pepai.config import load_config, results_dir

ROAD_USERS = {1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 6: "bus", 8: "truck"}
MODELS = ["frcnn_r50_fpn", "yolov10s", "yolov10x"]
PRECISIONS = ["fp32", "fp16", "int8", "fp8"]
THRESHOLD = 0.5


def operating_point(ev, cat_ids, area_label):
    """Recall and precision at score >= THRESHOLD and IoU 0.5 from COCOeval.evalImgs."""
    a = ev.params.areaRngLbl.index(area_label)
    tp = fp = n_gt = 0
    for e in ev.evalImgs:
        if e is None or e["category_id"] not in cat_ids or e["aRng"] != ev.params.areaRng[a]:
            continue
        scores = np.array(e["dtScores"])
        keep = scores >= THRESHOLD
        dt_m = np.array(e["dtMatches"])[0][keep] if len(scores) else np.zeros(0)
        dt_ig = np.array(e["dtIgnore"])[0][keep] if len(scores) else np.zeros(0, bool)
        gt_ig = np.array(e["gtIgnore"]).astype(bool)
        tp += int(np.sum((dt_m > 0) & ~dt_ig))
        fp += int(np.sum((dt_m == 0) & ~dt_ig))
        n_gt += int(np.sum(~gt_ig))
    return tp / max(n_gt, 1), tp / max(tp + fp, 1), n_gt


if __name__ == "__main__":
    cfg = load_config()
    coco, ids = data.coco_val_ids(cfg)
    det_dir = results_dir(cfg, "detections")
    rows = []
    for m in MODELS:
        for p in PRECISIONS:
            f = det_dir / f"{m}_{p}.json"
            if not f.exists():
                continue
            dets = json.loads(f.read_text())
            ev = COCOeval(coco, coco.loadRes(dets), "bbox")
            ev.params.imgIds = ids
            with contextlib.redirect_stdout(io.StringIO()):
                ev.evaluate()
                ev.accumulate()
            prec = ev.eval["precision"]                   # [T, R, K, A, M]
            k_index = {c: i for i, c in enumerate(ev.params.catIds)}
            row = {"model": m, "precision": p}
            for c, name in ROAD_USERS.items():
                pr = prec[:, :, k_index[c], 0, -1]
                row[f"AP_{name}"] = float(np.mean(pr[pr > -1]))
                pr50 = prec[0, :, k_index[c], 0, -1]
                row[f"AP50_{name}"] = float(np.mean(pr50[pr50 > -1]))
            for label, area in (("all", "all"), ("ml", "medium"), ("large", "large")):
                if label == "ml":
                    r_m, p_m, n_m = operating_point(ev, set(ROAD_USERS), "medium")
                    r_l, p_l, n_l = operating_point(ev, set(ROAD_USERS), "large")
                    # detections are assigned to area ranges by their matched ground truth; pool the counts
                    row["recall_road_ml"] = (r_m * n_m + r_l * n_l) / (n_m + n_l)
                    continue
                r, pr_, n = operating_point(ev, set(ROAD_USERS), area)
                row[f"recall_road_{label}"], row[f"precision_road_{label}"], row[f"n_gt_{label}"] = r, pr_, n
            for c in (1, 3):
                r, pr_, _ = operating_point(ev, {c}, "all")
                row[f"recall_{ROAD_USERS[c]}"], row[f"precision_{ROAD_USERS[c]}"] = r, pr_
            rows.append(row)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    pd.DataFrame(rows).to_csv(results_dir(cfg, "tables") / "operating_point.csv", index=False)
