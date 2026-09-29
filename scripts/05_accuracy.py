"""Task accuracy per precision; detections are stored for the FP32-vs-INT8 disagreement analysis."""
import argparse
import json

import numpy as np
import pandas as pd

from pepai import data
from pepai.config import load_config, results_dir
from pepai.evaluate import classify, coco_map, detect_frcnn, detect_yolo

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--precisions", nargs="*")
    ap.add_argument("--limit", type=int, help="evaluate only the first N images (smoke test)")
    args = ap.parse_args()
    cfg = load_config()
    engine_dir = results_dir(cfg, "engines")
    det_dir = results_dir(cfg, "detections")
    table = results_dir(cfg, "tables") / "accuracy.csv"
    rows = pd.read_csv(table).to_dict("records") if table.exists() and not args.limit else []

    coco, ids = data.coco_val_ids(cfg, args.limit)
    _, cls_items = data.imagenetv2_split(cfg)
    if args.limit:
        cls_items = cls_items[:args.limit]

    for name in args.models or cfg["models"]:
        preds_fp32 = None
        for precision in args.precisions or cfg["precisions"]:
            rows = [r for r in rows if not (r["model"] == name and r["precision"] == precision)]
            row = {"model": name, "precision": precision, "n_images": len(ids)}
            if name == "frcnn_r50_fpn" or name.startswith("yolov10"):
                if name == "frcnn_r50_fpn":
                    dets = detect_frcnn(cfg, engine_dir / f"{name}_{precision}_dyn.engine", coco, ids)
                else:
                    dets = detect_yolo(cfg, engine_dir / f"{name}_{precision}_bs1.engine", coco, ids)
                if not args.limit:
                    (det_dir / f"{name}_{precision}.json").write_text(json.dumps(dets))
                row.update(coco_map(coco, dets, ids))
            else:
                bs = max(cfg["benchmark"]["batch_sizes"])
                preds, labels = classify(engine_dir / f"{name}_{precision}_bs{bs}.engine", cls_items, bs)
                if not args.limit:      # per-image predictions for bootstrap confidence intervals
                    np.savez(det_dir / f"{name}_{precision}_cls.npz", preds=preds, labels=labels)
                if precision == "fp32":
                    preds_fp32 = preds
                row.update({"n_images": len(labels), "top1": float((preds == labels).mean())})
                if preds_fp32 is not None:
                    row["top1_agreement_with_fp32"] = float((preds == preds_fp32).mean())
            rows.append(row)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
            if not args.limit:
                pd.DataFrame(rows).to_csv(table, index=False)
