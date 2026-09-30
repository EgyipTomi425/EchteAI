"""Deployment-artefact ablation for the YOLOv10 head (GPU): quantizer placement of TopK and DFL decoder.

The configuration keeps the TopK of the NMS-free head and the distribution-focal-loss (DFL) box decoder out of
quantization (configs/default.yaml: exclude_op_types, exclude_nodes). This script re-quantizes YOLOv10-S with
(i) both quantized, the default placement of the toolchain, and (ii) the TopK kept in high precision but the
DFL decoder quantized, with the same calibration data, and evaluates both INT8 graphs (FP16 and FP32 fallback)
on COCO val2017. The main results correspond to (iii) both kept in high precision.
"""
import argparse

import pandas as pd

from pepai import data
from pepai.config import load_config, results_dir
from pepai.evaluate import coco_map, detect_yolo
from pepai.models import SPECS
from pepai.quant import calibration_inputs, excluded_op_types, int8_with_fp16, to_int8
from pepai.trt import build_engine

VARIANTS = {"TopK quantized": {"ops": [], "nodes": []},
            "TopK excluded, DFL quantized": {"ops": "config", "nodes": []}}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov10s")
    ap.add_argument("--out", default="ablation_topk.csv")
    args = ap.parse_args()
    cfg = load_config()
    spec = SPECS[args.model]
    coco, ids = data.coco_val_ids(cfg)
    calib = calibration_inputs(cfg, args.model)
    out_dir = results_dir(cfg, "ablation")
    table = results_dir(cfg, "tables") / args.out
    rows = pd.read_csv(table).to_dict("records") if table.exists() else []
    for variant, v in VARIANTS.items():
        tag = variant.lower().replace(",", "").replace(" ", "_")
        q32, q16 = out_dir / f"{args.model}_{tag}_int8fp32.onnx", out_dir / f"{args.model}_{tag}_int8.onnx"
        ops = excluded_op_types(cfg, args.model) if v["ops"] == "config" else v["ops"]
        if not q32.exists():
            to_int8(results_dir(cfg, "onnx") / f"{args.model}_fp32.onnx", q32, calib, method="entropy",
                    nodes_to_exclude=v["nodes"], op_types_to_exclude=ops)
            int8_with_fp16(q32, q16)
        for precision, onnx_path in (("int8", q16), ("int8fp32", q32)):
            if any(r["variant"] == variant and r["precision"] == precision for r in rows):
                continue
            engine = out_dir / f"{args.model}_{tag}_{precision}_bs1.engine"
            build_engine(onnx_path, engine, {spec.input_name: ((1, *spec.bench_shape),) * 3},
                         timing_cache=results_dir(cfg, "engines") / "timing.cache")
            row = {"model": args.model, "precision": precision, "n_images": len(ids),
                   **coco_map(coco, detect_yolo(cfg, engine, coco, ids), ids), "variant": variant}
            engine.unlink()
            rows.append(row)
            pd.DataFrame(rows).to_csv(table, index=False)
            print({k: (round(x, 4) if isinstance(x, float) else x) for k, x in row.items()}, flush=True)
