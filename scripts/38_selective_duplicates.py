"""Duplicate suppression of YOLOv10-X under PEP-AI-guided selective precision (GPU).

Builds the deployment engine of the selective-precision variants (k convolutions in FP16, PEP-AI ranking) and
measures, as 33_duplicates.py does for the base engines, the share of duplicate detections above the operating
threshold and the mAP without and with class-wise NMS.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd

from pepai import data
from pepai.config import load_config, results_dir
from pepai.evaluate import detect_yolo
from pepai.models import SPECS
from pepai.trt import build_engine

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov10x")
    ap.add_argument("--ks", nargs="*", type=int, default=[20])
    args = ap.parse_args()
    cfg = load_config()
    spec33 = importlib.util.spec_from_file_location("dup", Path(__file__).with_name("33_duplicates.py"))
    dup = importlib.util.module_from_spec(spec33)
    spec33.loader.exec_module(dup)
    coco, ids = data.coco_val_ids(cfg)
    spec = SPECS[args.model]
    table = results_dir(cfg, "tables") / "selective_duplicates.csv"
    rows = pd.read_csv(table).to_dict("records") if table.exists() else []
    for k in args.ks:
        if any(r["k"] == k and r["model"] == args.model for r in rows):
            continue
        onnx_path = next(results_dir(cfg, "selective").glob(f"{args.model}_default_pepai_k{k}_*_int8.onnx"))
        engine = results_dir(cfg, "selective") / f"{args.model}_pepai_k{k}_dup_bs1.engine"
        build_engine(onnx_path, engine, {spec.input_name: ((1, *spec.bench_shape),) * 3},
                     timing_cache=results_dir(cfg, "engines") / "timing.cache")
        dets = detect_yolo(cfg, engine, coco, ids)
        engine.unlink()
        (results_dir(cfg, "detections") / f"{args.model}_selective_k{k}.json").write_text(json.dumps(dets))
        share, n = dup.duplicate_share(dets)
        mAP, large = dup.evaluate(coco, ids, dets)
        mAP_nms, large_nms = dup.evaluate(coco, ids, dup.nms(dets, dup.DUP_IOU))
        rows.append({"model": args.model, "k": k, "dets_above_threshold": n, "duplicate_share": share,
                     "mAP": mAP, "mAP_nms": mAP_nms, "mAP_large": large, "mAP_large_nms": large_nms})
        pd.DataFrame(rows).to_csv(table, index=False)
        print(rows[-1], flush=True)
