"""Order-independent entropy calibration (GPU).

The entropy calibrator of the ONNX PTQ toolchain (ModelOpt on ONNX Runtime) builds its histogram incrementally,
one calibration batch (here: one image) at a time: the first batch fixes the bin width (128 bins over its own
range), later batches only append bins, and the threshold search starts at the range of the first batch. The
resulting scales therefore depend on which image comes first (29_calibration_variability.py,
30_calibration_scales.py).

This script implements entropy calibration as originally defined for TensorRT (Migacz, GTC 2017): pass 1 collects
the global maximum of |x| of every quantized activation over the whole calibration set; pass 2 accumulates a
2048-bin histogram of |x| over [0, max]; the threshold minimises the KL divergence between the clipped reference
distribution and its 128-level quantization over all candidates of 128..2048 bins, using NVIDIA's reference
implementation of this search (ModelOpt, which also neutralises the zero bin). Both passes are sums and maxima
over images, so the result does not depend on their order. The thresholds replace the activation scales of the
full-set INT8 graph (weights, placement and exclusions unchanged), and the deployment engine is evaluated as in
the main results.
"""
import json
import argparse

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
from onnx import numpy_helper
import ast
from collections import Counter
from pathlib import Path

import modelopt
from scipy.stats import entropy

from pepai import data
from pepai.config import load_config, results_dir
from pepai.evaluate import classify, coco_map, detect_frcnn, detect_yolo
from pepai.models import SPECS, engine_shapes
from pepai.quant import calibration_inputs, int8_with_fp16
from pepai.trt import build_engine

BINS, LEVELS = 2048, 128
ORDER = ["efficientnet_b0", "densenet121", "yolov10s", "yolov10x", "frcnn_r50_fpn"]


def activation_quantizers(graph):
    """Quantized activation tensor -> names of the scale initializers of its Q and DQ nodes."""
    init = {i.name for i in graph.initializer}
    consumers = {}
    for n in graph.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    out = {}
    for n in graph.node:
        if n.op_type == "QuantizeLinear" and n.input[0] not in init:
            names = {n.input[1]}
            names.update(dq.input[1] for dq in consumers.get(n.output[0], []) if dq.op_type == "DequantizeLinear")
            out[n.input[0]] = names
    return out


def _reference_entropy_search():
    """NVIDIA's reference KL search (ModelOpt, modelopt/torch/quantization/calib/histogram.py), loaded from the
    installed source; importing modelopt.torch itself pulls in optional dependencies that are not installed."""
    src = Path(modelopt.__file__).parent / "torch" / "quantization" / "calib" / "histogram.py"
    fn = next(n for n in ast.parse(src.read_text()).body
              if isinstance(n, ast.FunctionDef) and n.name == "_compute_amax_entropy")
    ns = {"np": np, "Counter": Counter, "entropy": entropy, "torch": torch}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(src), "exec"), ns)
    return ns["_compute_amax_entropy"]


_compute_amax_entropy = _reference_entropy_search()


def kl_threshold(hist, amax):
    """Threshold minimising the KL divergence, computed with NVIDIA's reference implementation (ModelOpt), which
    replaces the zero bin by its neighbour so that the spike of zeros after ReLU/SiLU does not dominate."""
    edges = np.linspace(0, amax, BINS + 1)
    return float(_compute_amax_entropy(hist.copy(), edges, num_bits=8, unsigned=False, start_bin=LEVELS))


def run_outputs(session, calib, names):
    """Values of the named tensors for every calibration image; graph inputs are taken from the image itself."""
    input_name = session.get_inputs()[0].name
    fetch = [t for t in names if t != input_name]
    for x in (calib if isinstance(calib, list) else (calib[i:i + 1] for i in range(len(calib)))):
        got = dict(zip(fetch, session.run(fetch, {input_name: x}))) if fetch else {}
        yield [x if t == input_name else got[t] for t in names]


def global_entropy_thresholds(fp32_path, tensors, calib, chunk):
    model = onnx.load(str(fp32_path))
    known = {vi.name for vi in model.graph.value_info} | {o.name for o in model.graph.output} | \
        {i.name for i in model.graph.input} | {o for n in model.graph.node for o in n.output}
    present = [t for t in tensors if t in known]
    existing = {o.name for o in model.graph.output} | {i.name for i in model.graph.input}
    for t in present:
        if t not in existing:
            model.graph.output.append(onnx.helper.make_empty_tensor_value_info(t))
    session = ort.InferenceSession(model.SerializeToString(), providers=["CUDAExecutionProvider"])
    thresholds = {}
    for k in range(0, len(present), chunk):
        names = present[k:k + chunk]
        amax = {t: 0.0 for t in names}
        for outs in run_outputs(session, calib, names):                          # pass 1: global range
            for t, o in zip(names, outs):
                amax[t] = max(amax[t], float(np.abs(o).max()))
        hist = {t: torch.zeros(BINS, dtype=torch.float64, device="cuda") for t in names}
        for outs in run_outputs(session, calib, names):                          # pass 2: histogram
            for t, o in zip(names, outs):
                if amax[t] > 0:
                    v = torch.from_numpy(np.abs(o).ravel()).cuda()
                    hist[t] += torch.histc(v.float(), bins=BINS, min=0, max=amax[t]).double()
        for t in names:
            thresholds[t] = kl_threshold(hist[t].cpu().numpy(), amax[t]) if amax[t] > 0 else 0.0
        print(f"  thresholds {k + len(names)}/{len(present)}", flush=True)
    return thresholds, len(tensors) - len(present)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=ORDER)
    ap.add_argument("--chunk", type=int, default=48)
    ap.add_argument("--variant", choices=["global"], default="global")
    args = ap.parse_args()
    cfg = load_config()
    out = results_dir(cfg, "calib_global")
    table = results_dir(cfg, "tables") / "global_entropy.csv"
    rows = pd.read_csv(table).to_dict("records") if table.exists() else []
    done = {(r["model"], r.get("variant") if isinstance(r.get("variant"), str) else "global") for r in rows}
    coco, ids = data.coco_val_ids(cfg)
    _, cls_items = data.imagenetv2_split(cfg)
    for name in args.models:
        if (name, args.variant) in done:
            continue
        spec = SPECS[name]
        q32_full = results_dir(cfg, "onnx") / f"{name}_int8fp32.onnx"
        qg = onnx.load(str(q32_full))
        quant = activation_quantizers(qg.graph)
        calib = calibration_inputs(cfg, name)
        cache = out / f"{name}_global_thresholds.json"
        if cache.exists():
            cached = json.loads(cache.read_text())
            thr, missing = cached["thresholds"], cached["missing"]
        else:
            thr, missing = global_entropy_thresholds(results_dir(cfg, "onnx") / f"{name}_fp32.onnx", list(quant),
                                                     calib, args.chunk)
            cache.write_text(json.dumps({"thresholds": thr, "missing": missing}))
        ratios = []
        for t, names in quant.items():
            if t not in thr or thr[t] <= 0:
                continue
            for i in qg.graph.initializer:
                if i.name in names:
                    old = numpy_helper.to_array(i)
                    new = np.array(thr[t] / 127, dtype=old.dtype)
                    ratios.append(float(new) / float(old))
                    i.CopyFrom(numpy_helper.from_array(new, i.name))
        q32 = out / f"{name}_{args.variant}_int8fp32.onnx"
        q16 = out / f"{name}_{args.variant}_int8.onnx"
        onnx.save(qg, str(q32))
        int8_with_fp16(q32, q16)
        if name == "frcnn_r50_fpn":
            engine, shapes = out / f"{name}_{args.variant}_int8_dyn.engine", engine_shapes(spec)
        elif spec.task == "det":
            engine, shapes = out / f"{name}_{args.variant}_int8_bs1.engine", {spec.input_name: ((1, *spec.bench_shape),) * 3}
        else:
            engine, shapes = out / f"{name}_{args.variant}_int8_bs8.engine", {spec.input_name: ((8, *spec.bench_shape),) * 3}
        build_engine(q16, engine, shapes, timing_cache=results_dir(cfg, "engines") / "timing.cache")
        row = {"model": name, "variant": args.variant, "n_quantizers": len(quant), "n_missing": missing,
               "median_scale_ratio": float(np.median(ratios)), "share_scale_changed_25pct":
               float(np.mean(np.abs(np.log2(ratios)) > np.log2(1.25)))}
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
