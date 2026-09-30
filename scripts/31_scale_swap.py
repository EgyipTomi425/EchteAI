"""Causal test of calibration-sample sensitivity by swapping activation scales (GPU).

Starting from the full-set INT8 graph of a model, the activation scales of selected quantizers are replaced by
their values from a half-set calibration (29_calibration_variability.py) and the deployment engine is
evaluated. Variants: only the quantizer in front of the top-ranked PEP-AI layer ("top"), all quantizers except
that one ("rest"), and all quantizers ("all", which reproduces the half-set graph). If "top" alone moves the
accuracy most of the way, the sensitivity of INT8 accuracy to the calibration sample is concentrated in the
layer that the activation analysis ranks first.
"""
import argparse

import numpy as np
import onnx
import pandas as pd
from onnx import numpy_helper

from pepai import data
from pepai.config import load_config, results_dir
from pepai.evaluate import classify, coco_map, detect_yolo
from pepai.models import SPECS
from pepai.quant import int8_with_fp16
from pepai.trt import build_engine


def scale_tensors(graph):
    """Map: quantized activation tensor -> names of the scale initializers of its Q and DQ nodes."""
    init = {i.name for i in graph.initializer}
    consumers = {}
    for n in graph.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    out = {}
    for n in graph.node:
        if n.op_type == "QuantizeLinear" and n.input[0] not in init:
            names = {n.input[1]}
            for dq in consumers.get(n.output[0], []):
                if dq.op_type == "DequantizeLinear":
                    names.add(dq.input[1])
            out[n.input[0]] = names
    return out


def scale_value(graph, tensor):
    names = scale_tensors(graph)[tensor]
    vals = {i.name: numpy_helper.to_array(i) for i in graph.initializer if i.name in names}
    return float(next(iter(vals.values())))


def swap(full_path, half_path, tensors, out_path):
    full = onnx.load(str(full_path))
    half = onnx.load(str(half_path), load_external_data=False)
    targets = scale_tensors(full.graph)
    for t in tensors:
        new = scale_value(half.graph, t)
        for i in full.graph.initializer:
            if i.name in targets[t]:
                i.CopyFrom(numpy_helper.from_array(np.array(new, dtype=numpy_helper.to_array(i).dtype), i.name))
    onnx.save(full, str(out_path))


def top_quantizer(cfg, name):
    """Input tensor of the convolution ranked first by the iterative PEP-AI analysis."""
    sel = pd.read_csv(results_dir(cfg, "tables") / "selective.csv")
    row = sel[(sel.model == name) & (sel.strategy == "pepai_iter") & (sel.k == 1)].iloc[0]
    conv = row.excluded.split(";")[0]
    g = onnx.load(str(results_dir(cfg, "onnx") / f"{name}_fp32.onnx"), load_external_data=False).graph
    return next(n.input[0] for n in g.node if n.name == conv)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    cfg = load_config()
    spec = SPECS[args.model]
    d = results_dir(cfg, "calib_variability")
    full_p = results_dir(cfg, "onnx") / f"{args.model}_int8fp32.onnx"
    half_p = d / f"{args.model}_s{args.seed}_int8fp32.onnx"
    full_g = onnx.load(str(full_p), load_external_data=False).graph
    all_t = list(scale_tensors(full_g))
    top = top_quantizer(cfg, args.model)
    assert top in all_t, top
    variants = {"top": [top], "rest": [t for t in all_t if t != top], "all": all_t}
    table = results_dir(cfg, "tables") / "scale_swap.csv"
    rows = pd.read_csv(table).to_dict("records") if table.exists() else []
    done = {(r["model"], r["seed"], r["variant"]) for r in rows}
    if spec.task == "cls":
        _, items = data.imagenetv2_split(cfg)
    else:
        coco, ids = data.coco_val_ids(cfg)
    for v, tensors in variants.items():
        if (args.model, args.seed, v) in done:
            continue
        q32 = d / f"{args.model}_s{args.seed}_swap_{v}_int8fp32.onnx"
        q16 = d / f"{args.model}_s{args.seed}_swap_{v}_int8.onnx"
        swap(full_p, half_p, tensors, q32)
        int8_with_fp16(q32, q16)
        bs = 8 if spec.task == "cls" else 1
        engine = d / f"{args.model}_s{args.seed}_swap_{v}_bs{bs}.engine"
        build_engine(q16, engine, {spec.input_name: ((bs, *spec.bench_shape),) * 3},
                     timing_cache=results_dir(cfg, "engines") / "timing.cache")
        row = {"model": args.model, "seed": args.seed, "variant": v, "n_swapped": len(tensors),
               "top_tensor": top, "scale_full_top": scale_value(full_g, top),
               "scale_half_top": scale_value(onnx.load(str(half_p), load_external_data=False).graph, top)}
        if spec.task == "cls":
            preds, labels = classify(engine, items, bs)
            row["top1"] = float((preds == labels).mean())
        else:
            row.update(coco_map(coco, detect_yolo(cfg, engine, coco, ids), ids))
        engine.unlink()
        rows.append(row)
        pd.DataFrame(rows).to_csv(table, index=False)
        print({k: (round(x, 4) if isinstance(x, float) else x) for k, x in row.items()}, flush=True)
