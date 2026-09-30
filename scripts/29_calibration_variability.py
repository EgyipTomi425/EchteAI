"""Sensitivity of INT8 accuracy to the calibration sample (GPU).

For every model, three disjoint-seeded random halves of the calibration set (256 of the 512 COCO train2017
images, 500 of the 1 000 ImageNetV2 calibration images) are used to calibrate the default INT8 configuration
(entropy calibration, same exclusions); the resulting deployment engines (FP16 fallback) are evaluated on
the full evaluation sets. The spread across seeds quantifies how much of the reported INT8 accuracy depends
on the particular calibration sample.
"""
import argparse
import random

import numpy as np
import pandas as pd

from pepai import data
from pepai.config import load_config, results_dir
from pepai.evaluate import classify, coco_map, detect_frcnn, detect_yolo
from pepai.models import SPECS, engine_shapes
from pepai.quant import calibration_inputs, excluded_nodes, excluded_op_types, int8_with_fp16, to_int8
from pepai.trt import build_engine

ORDER = ["efficientnet_b0", "densenet121", "yolov10s", "frcnn_r50_fpn", "yolov10x"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=ORDER)
    ap.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3])
    ap.add_argument("--fraction", type=float, default=0.5)
    args = ap.parse_args()
    cfg = load_config()
    out = results_dir(cfg, "calib_variability")
    table = results_dir(cfg, "tables") / "calibration_variability.csv"
    rows = pd.read_csv(table).to_dict("records") if table.exists() else []
    done = {(r["model"], r["seed"], r["n_calib"]) for r in rows}
    coco, ids = data.coco_val_ids(cfg)
    _, cls_items = data.imagenetv2_split(cfg)
    for name in args.models:
        spec = SPECS[name]
        calib = None
        for seed in args.seeds:
            if calib is None:
                calib = calibration_inputs(cfg, name)
            n = len(calib)
            if (name, seed, int(n * args.fraction)) in done:
                continue
            tag = f"s{seed}" if args.fraction == 0.5 else f"f{round(100 * args.fraction)}_s{seed}"
            idx = sorted(random.Random(seed).sample(range(n), int(n * args.fraction)))
            subset = [calib[i] for i in idx] if isinstance(calib, list) else calib[idx]
            q32 = out / f"{name}_{tag}_int8fp32.onnx"
            q16 = out / f"{name}_{tag}_int8.onnx"
            to_int8(results_dir(cfg, "onnx") / f"{name}_fp32.onnx", q32, subset, method="entropy",
                    nodes_to_exclude=excluded_nodes(cfg, name), op_types_to_exclude=excluded_op_types(cfg, name))
            int8_with_fp16(q32, q16)
            if name == "frcnn_r50_fpn":
                engine = out / f"{name}_{tag}_int8_dyn.engine"
                shapes = engine_shapes(spec)
            elif spec.task == "det":
                engine = out / f"{name}_{tag}_int8_bs1.engine"
                shapes = {spec.input_name: ((1, *spec.bench_shape),) * 3}
            else:
                engine = out / f"{name}_{tag}_int8_bs8.engine"
                shapes = {spec.input_name: ((8, *spec.bench_shape),) * 3}
            build_engine(q16, engine, shapes, timing_cache=results_dir(cfg, "engines") / "timing.cache")
            row = {"model": name, "seed": seed, "n_calib": len(idx)}
            if spec.task == "cls":
                preds, labels = classify(engine, cls_items, 8)
                row["top1"] = float((preds == labels).mean())
            else:
                det = detect_frcnn if name == "frcnn_r50_fpn" else detect_yolo
                row.update(coco_map(coco, det(cfg, engine, coco, ids), ids))
            engine.unlink()
            rows.append(row)
            pd.DataFrame(rows).to_csv(table, index=False)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
