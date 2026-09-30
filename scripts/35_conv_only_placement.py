"""EfficientNet-B0 with quantizers on convolution and matrix-multiplication inputs only (GPU).

The default ModelOpt placement also quantizes the tensors of the squeeze-and-excitation blocks (sigmoid gates,
global pooling, the gating multiplication) and the residual additions. Published INT8 results for EfficientNet
(Wu et al., 2020) quantize only the inputs of convolutions and GEMMs. This script builds that placement from the
same FP32 model and calibration set and evaluates it with (i) the toolchain's entropy calibration and (ii) the
order-independent entropy thresholds of 34_global_entropy.py, so that the gap to the literature can be attributed
to placement and calibration separately.
"""
import json

import numpy as np
import onnx
import pandas as pd
from onnx import numpy_helper

from pepai import data
from pepai.config import load_config, results_dir
from pepai.evaluate import classify
from pepai.models import SPECS
from pepai.quant import calibration_inputs, int8_with_fp16, to_int8
from pepai.trt import build_engine

NAME = "efficientnet_b0"
NON_CONV_OPS = ["Mul", "Add", "Sigmoid", "GlobalAveragePool", "AveragePool", "MaxPool", "Concat"]


def activation_quantizers(graph):
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


if __name__ == "__main__":
    cfg = load_config()
    spec = SPECS[NAME]
    out = results_dir(cfg, "calib_global")
    table = results_dir(cfg, "tables") / "conv_only_placement.csv"
    rows = pd.read_csv(table).to_dict("records") if table.exists() else []
    done = {r["variant"] for r in rows}
    _, items = data.imagenetv2_split(cfg)
    q32_tool = out / f"{NAME}_convonly_toolchain_int8fp32.onnx"
    if not q32_tool.exists():
        to_int8(results_dir(cfg, "onnx") / f"{NAME}_fp32.onnx", q32_tool, calibration_inputs(cfg, NAME),
                method="entropy", op_types_to_exclude=NON_CONV_OPS)
    graph = onnx.load(str(q32_tool))
    quant = activation_quantizers(graph.graph)
    thresholds = json.loads((out / f"{NAME}_global_thresholds.json").read_text())["thresholds"]
    for variant in ("toolchain", "global"):
        if variant in done:
            continue
        q32 = out / f"{NAME}_convonly_{variant}_int8fp32.onnx"
        n_missing = 0
        if variant == "global":
            g = onnx.load(str(q32_tool))
            for t, names in quant.items():
                if t not in thresholds:
                    n_missing += 1
                    continue
                for i in g.graph.initializer:
                    if i.name in names:
                        old = numpy_helper.to_array(i)
                        i.CopyFrom(numpy_helper.from_array(np.array(thresholds[t] / 127, dtype=old.dtype), i.name))
            onnx.save(g, str(q32))
        q16 = out / f"{NAME}_convonly_{variant}_int8.onnx"
        int8_with_fp16(q32, q16)
        engine = out / f"{NAME}_convonly_{variant}_int8_bs8.engine"
        build_engine(q16, engine, {spec.input_name: ((8, *spec.bench_shape),) * 3},
                     timing_cache=results_dir(cfg, "engines") / "timing.cache")
        preds, labels = classify(engine, items, 8)
        engine.unlink()
        row = {"model": NAME, "variant": variant, "n_quantizers": len(quant), "n_missing": n_missing,
               "top1": float((preds == labels).mean())}
        rows.append(row)
        pd.DataFrame(rows).to_csv(table, index=False)
        print(row, flush=True)
