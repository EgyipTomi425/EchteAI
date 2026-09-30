"""Duplicate detections of the NMS-free YOLOv10 head under quantization (CPU only, from stored detections).

YOLOv10 is trained with a one-to-one assignment, so that at inference its head is expected to emit a single box
per object without non-maximum suppression (NMS). This script measures, per engine, the share of detections
above the operating threshold that duplicate a higher-scoring same-class box (IoU >= 0.7), and the COCO mAP
after class-wise NMS (IoU 0.7) is appended as post-processing. If NMS restores most of the INT8 loss, the loss
lies in the implicit duplicate suppression of the one-to-one head rather than in localisation or recognition.
"""
import contextlib
import io
import json

import numpy as np
import pandas as pd
import torch
from pycocotools.cocoeval import COCOeval
from torchvision.ops import batched_nms

from pepai import data
from pepai.config import load_config, results_dir

THRESHOLD, DUP_IOU = 0.5, 0.7


def evaluate(coco, ids, dets):
    ev = COCOeval(coco, coco.loadRes(dets), "bbox")
    ev.params.imgIds = ids
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return ev.stats[0], ev.stats[5]


def nms(dets, iou):
    by_img = {}
    for d in dets:
        by_img.setdefault(d["image_id"], []).append(d)
    out = []
    for v in by_img.values():
        b = torch.tensor([d["bbox"] for d in v], dtype=torch.float32)
        b[:, 2:] += b[:, :2]
        keep = batched_nms(b, torch.tensor([d["score"] for d in v]), torch.tensor([d["category_id"] for d in v]), iou)
        out += [v[i] for i in keep.tolist()]
    return out


def duplicate_share(dets):
    kept = [d for d in dets if d["score"] >= THRESHOLD]
    after = nms(kept, DUP_IOU)
    return 1 - len(after) / max(len(kept), 1), len(kept)


if __name__ == "__main__":
    cfg = load_config()
    coco, ids = data.coco_val_ids(cfg)
    rows = []
    for m in ("yolov10s", "yolov10x"):
        for p in ("fp32", "fp16", "int8", "int8fp32", "fp8"):
            f = results_dir(cfg, "detections") / f"{m}_{p}.json"
            if not f.exists():
                continue
            dets = json.loads(f.read_text())
            share, n = duplicate_share(dets)
            mAP, large = evaluate(coco, ids, dets)
            mAP_nms, large_nms = evaluate(coco, ids, nms(dets, DUP_IOU))
            row = {"model": m, "precision": p, "dets_above_threshold": n, "duplicate_share": share,
                   "mAP": mAP, "mAP_nms": mAP_nms, "mAP_large": large, "mAP_large_nms": large_nms}
            rows.append(row)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    pd.DataFrame(rows).to_csv(results_dir(cfg, "tables") / "duplicates.csv", index=False)
