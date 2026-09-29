"""Per-image head-input deviation vs FP32/INT8 detection disagreement on the full COCO val2017.

Both quantities come from the same pair of engines (strict FP32 and INT8 QDQ with FP32 fallback)
in which only the head-input feature tensors are additionally marked as outputs.
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from pepai import data
from pepai.agreement import box_iou, image_disagreement
from pepai.config import load_config, results_dir
from pepai.risk import PairRunner


def ground_truth(coco, img_id):
    anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=False))
    boxes = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2],
                       a["bbox"][1] + a["bbox"][3]] for a in anns]).reshape(-1, 4)
    return boxes, np.array([a["category_id"] for a in anns])


def detection_level(runner, df, dq, img_id, thr, tolerance=0.2):
    """One row per FP32 detection at the operating threshold: does it flip under INT8?"""
    keep = df["scores"] >= thr
    boxes, labels, scores = df["boxes"][keep], df["labels"][keep], df["scores"][keep]
    rows = []
    iou = box_iou(boxes, dq["boxes"]) if len(boxes) and len(dq["boxes"]) else np.zeros((len(boxes), 0))
    for i in range(len(boxes)):
        same = dq["labels"] == labels[i] if len(dq["boxes"]) else np.zeros(0, bool)
        def matched(q_thr):
            ok = same & (dq["scores"] >= q_thr) if len(dq["boxes"]) else same
            return bool((iou[i][ok] >= 0.5).any()) if ok.any() else False
        w, h = boxes[i][2] - boxes[i][0], boxes[i][3] - boxes[i][1]
        rows.append({"image": img_id, "score": float(scores[i]), "log_area": float(np.log(max(w * h, 1.0))),
                     "local_mre": runner.local_deviation(boxes[i]),
                     "flip": int(not matched(thr)), "flip_tol": int(not matched(thr - tolerance))})
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["frcnn_r50_fpn", "yolov10s", "yolov10x"])
    ap.add_argument("--quant", default="int8fp32")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--score-thr", type=float, default=0.5)
    args = ap.parse_args()
    cfg = load_config()
    coco, ids = data.coco_val_ids(cfg, args.limit)
    out_dir = results_dir(cfg, "risk")

    for name in args.models:
        runner = PairRunner(cfg, name, args.quant)
        rows, shifts, det_rows = [], [], []
        for img_id in tqdm(ids, desc=name):
            img = data.load_rgb(data.coco_val_path(cfg, coco, img_id))
            metrics, df, dq = runner(img)
            dis = image_disagreement(df, dq, *ground_truth(coco, img_id), score_thr=args.score_thr)
            shifts.append({"image": img_id, "centre": dis.pop("centre_shift_px"),
                           "bottom": dis.pop("bottom_shift_px"), "img_h": img.size[1]})
            rows.append({"image": img_id, **metrics, **dis})
            det_rows += detection_level(runner, df, dq, img_id, args.score_thr)

        tag = f"{name}_{args.quant}" + (f"_n{args.limit}" if args.limit else "")
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / f"{tag}.csv", index=False)
        pd.DataFrame(det_rows).to_csv(out_dir / f"{tag}_detections.csv", index=False)
        (out_dir / f"{tag}_shifts.json").write_text(json.dumps(shifts))
        print(f"{name}: images with disagreement {df.error.mean():.3f}, "
              f"median head-input MRE_proj {df.mre_proj.median():.4f}", flush=True)
        del runner
        torch.cuda.empty_cache()
