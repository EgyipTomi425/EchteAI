"""Robust calibration: half of the 512 calibration images carry a random adverse-condition corruption.

Motivated by the robustness study, where the INT8 penalty grew under low contrast because activation
ranges were calibrated on clean images only. Everything else (placement, exclusions, calibration
method, engines) is identical to the default INT8 model, so the difference isolates the calibration data.
"""
import argparse
import hashlib
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image

from pepai import data, models
from pepai.config import load_config, results_dir
from pepai.corruptions import CONDITIONS, SEVERITIES, apply
from pepai.evaluate import coco_map, detect_frcnn, detect_yolo
from pepai.models import SPECS, engine_shapes
from pepai.quant import (calibration_inputs, excluded_nodes, excluded_op_types, int8_with_fp16,
                         release_output_qdq, to_int8)
from pepai.sensitivity import conv_sensitivity
from pepai.trt import build_engine

EVAL = [("clean", 0)] + [(c, s) for c in ("contrast", "fog", "dark", "motion_blur") for s in (3, 5)]


def robust_calibration_inputs(cfg, name, share=0.5):
    rng = random.Random(cfg["seed"])
    out = []
    frcnn = models.load_frcnn().cuda() if name == "frcnn_r50_fpn" else None
    for path in data.coco_calib_paths(cfg):
        img = data.load_rgb(path)
        if rng.random() < share:
            np.random.seed(rng.randrange(2 ** 31))
            img = Image.fromarray(apply(np.asarray(img), rng.choice(list(CONDITIONS)), rng.choice(SEVERITIES)))
        if frcnn is not None:
            out.append(models.frcnn_preprocess(frcnn, [img]).tensors.cpu().numpy().astype(np.float32))
        else:
            out.append(models.letterbox(img)[0].numpy().astype(np.float32))
    return out if frcnn is not None else np.stack(out)


def input_conv(cfg, name):
    """The convolution that consumes the network input (the image)."""
    import onnx
    g = onnx.load(str(results_dir(cfg, "onnx") / f"{name}_fp32.onnx"), load_external_data=False).graph
    inp = g.input[0].name
    consumers = {inp}
    for n in g.node:                    # follow pass-through ops (e.g. casts) to the first Conv
        if any(i in consumers for i in n.input):
            if n.op_type == "Conv":
                return n.name
            consumers.update(n.output)
    raise ValueError(name)


def top_ranked_node(cfg, name):
    """Most error-injecting convolution according to the label-free activation analysis."""
    act = results_dir(cfg, "activations")
    rank = conv_sensitivity(act / f"{name}_int8fp32.csv.gz", act / f"{name}_int8fp32_tensors.json",
                            results_dir(cfg, "onnx") / f"{name}_fp32.onnx",
                            results_dir(cfg, "onnx") / f"{name}_int8fp32.onnx")
    return rank.node.iloc[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["yolov10s", "yolov10x"])
    args = ap.parse_args()
    cfg = load_config()
    work = results_dir(cfg, "ablation")
    table = results_dir(cfg, "tables") / "robust_calibration.csv"
    rows = pd.read_csv(table).to_dict("records") if table.exists() else []
    for r in rows:
        if pd.isna(r.get("variant", float("nan"))):
            r["variant"] = "robustcalib"
        if pd.isna(r.get("excluded", float("nan"))):
            r["excluded"] = ""
    coco, ids = data.coco_val_ids(cfg, cfg["coco"]["n_activation"])
    for name in args.models:
        top, first = top_ranked_node(cfg, name), input_conv(cfg, name)
        # robustcalib: corrupted calibration images; pepai_top1: the top-ranked quantized conv in FP16;
        # input_fp16: the convolution reading the image kept in FP16 (tests input quantization).
        variants = [("robustcalib", robust_calibration_inputs, []), ("pepai_top1", calibration_inputs, [top])]
        if first != top:
            variants.append(("input_fp16", calibration_inputs, [first]))
        for variant, calib_fn, extra_nodes in variants:
            # File names carry the excluded nodes, so a different node choice never reuses old artefacts.
            stem = f"{name}_{variant}_" + hashlib.sha1(";".join(extra_nodes).encode()).hexdigest()[:8]
            q32, q16 = work / f"{stem}_int8fp32.onnx", work / f"{stem}_int8.onnx"
            if not q16.exists():
                to_int8(results_dir(cfg, "onnx") / f"{name}_fp32.onnx", q32, calib_fn(cfg, name),
                        op_types_to_exclude=excluded_op_types(cfg, name),
                        nodes_to_exclude=excluded_nodes(cfg, name) + extra_nodes)
                if extra_nodes:
                    release_output_qdq(q32, extra_nodes)
                int8_with_fp16(q32, q16)
            eng = work / f"{stem}_int8_bs1.engine"
            if not eng.exists():
                spec = SPECS[name]
                build_engine(q16, eng, engine_shapes(spec) if spec.dynamic_hw else {spec.input_name: ((1, *spec.bench_shape),) * 3})
            excluded_key = ";".join(extra_nodes)
            for cond, sev in EVAL:
                same = [r for r in rows if r["model"] == name and r["variant"] == variant
                        and r["condition"] == cond and r["severity"] == sev]
                if any(str(r.get("excluded", "") or "") == excluded_key for r in same):
                    continue
                rows = [r for r in rows if r not in same]      # stale rows from another node choice
                load = None
                if cond != "clean":
                    d = cfg["coco"]["root"] / "coco_c" / cond / str(sev)
                    load = lambda i, d=d: data.load_rgb(d / f"{i}.png")  # noqa: E731
                det = detect_frcnn if name == "frcnn_r50_fpn" else detect_yolo
                rows.append({"model": name, "variant": variant, "excluded": excluded_key, "condition": cond,
                             "severity": sev, **coco_map(coco, det(cfg, eng, coco, ids, load=load), ids)})
                pd.DataFrame(rows).to_csv(table, index=False)
                print(rows[-1], flush=True)
        torch.cuda.empty_cache()
