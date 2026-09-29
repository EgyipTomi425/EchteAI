"""Layer sensitivity from the activation analysis: how much deviation each quantized node injects."""
import json
from pathlib import Path

import onnx
import pandas as pd
import torch

from pepai import debug
from pepai.activations import analysed_tensors, load_activation_table, resolved_inputs, tensor_metrics
from pepai.config import results_dir
from pepai.inputs import analysis_inputs
from pepai.models import SPECS, engine_shapes
from pepai.trt import TRTModel, build_engine


def quantized_nodes(quant_onnx):
    """Nodes that actually run quantized: at least one input comes from a DequantizeLinear."""
    graph = onnx.load(str(quant_onnx), load_external_data=False).graph
    dq = {o for n in graph.node if n.op_type == "DequantizeLinear" for o in n.output}
    return {n.name for n in graph.node if any(i in dq for i in n.input)}


def rank_convs(sqnr, infos, resolved=None, candidates=None):
    """Median (over images) SQNR drop across every Conv node: SQNR(input) - SQNR(output).

    sqnr: DataFrame images x tensors (dB); infos: dicts with name/op/node/order/inputs.
    A large drop means the node itself (its quantized input/weights) injects most of the error,
    rather than merely propagating upstream error. The network input is an 8-bit image, so its
    INT8 quantization is nearly lossless; the first layer is flagged only below 40 dB output SQNR.
    """
    rows = []
    for t in infos:
        if t["op"] != "Conv" or t["name"] not in sqnr:
            continue
        if candidates is not None and t["node"] not in candidates:
            continue            # not quantized: keeping it in high precision would change nothing
        out = sqnr[t["name"]]
        upstream = resolved.get(t["node"], t["inputs"]) if resolved is not None else t["inputs"]
        inputs = [i for i in upstream if i in sqnr and i != t["name"]]
        drop = sqnr[inputs].min(axis=1) - out if inputs else (40.0 - out).clip(lower=0)
        rows.append({"node": t["node"], "tensor": t["name"], "order": t["order"],
                     "sqnr_out_db": out.median(), "drop_db": drop.median()})
    return pd.DataFrame(rows).sort_values("drop_db", ascending=False).reset_index(drop=True)


def conv_sensitivity(activation_csv, tensors_json, fp32_onnx=None, quant_onnx=None):
    df = load_activation_table(activation_csv, usecols=["image", "tensor", "sqnr_db", "cosine", "abs_mean"])
    sqnr = df.pivot_table(index="image", columns="tensor", values="sqnr_db")
    infos = json.loads(Path(tensors_json).read_text())
    resolved = resolved_inputs(fp32_onnx, [t["name"] for t in infos]) if fp32_onnx else None
    candidates = quantized_nodes(quant_onnx) if quant_onnx else None
    return rank_convs(sqnr, infos, resolved, candidates)


@torch.no_grad()
def measure_and_rank(cfg, name, quant_onnx, n_images, work_dir, return_layers=False):
    """Activation analysis of an arbitrary quantized graph on n images, returning the Conv ranking
    (and, with return_layers, the per-tensor median SQNR / MRE_proj table)."""
    fp32 = results_dir(cfg, "onnx") / f"{name}_fp32.onnx"
    infos = analysed_tensors(fp32, quant_onnx)
    engine = Path(work_dir) / (Path(quant_onnx).stem + "_debug.engine")
    if not engine.exists():
        build_engine(quant_onnx, engine, engine_shapes(SPECS[name]), mark_outputs=[t.name for t in infos])
    ref = TRTModel(debug.ensure_engine(cfg, name, "fp32", "full"))
    qnt = TRTModel(engine)
    rows = []
    for key, x, _ in analysis_inputs(cfg, name, n_images):
        fo = {k: v.clone() for k, v in ref(images=x.to(ref.dtype("images"))).items()}
        qo = qnt(images=x.to(qnt.dtype("images")))
        for t in infos:
            m = tensor_metrics(fo[t.name], qo[t.name])
            rows.append({"image": key, "tensor": t.name, "op": t.op, "order": t.order,
                         "sqnr_db": m["sqnr_db"], "mre_proj": m.get("mre_proj")})
    engine.unlink()
    df = pd.DataFrame(rows)
    resolved = resolved_inputs(fp32, [t.name for t in infos])
    rank = rank_convs(df.pivot_table(index="image", columns="tensor", values="sqnr_db"), [vars(t) for t in infos],
                      resolved, quantized_nodes(quant_onnx))
    if not return_layers:
        return rank
    layers = df.groupby(["tensor", "op", "order"])[["sqnr_db", "mre_proj"]].median().reset_index()
    return rank, layers


def conv_nodes(fp32_onnx):
    """(node name, is depthwise) for every Conv of the FP32 graph."""
    out = []
    for n in onnx.load(str(fp32_onnx), load_external_data=False).graph.node:
        if n.op_type == "Conv":
            group = next((onnx.helper.get_attribute_value(a) for a in n.attribute if a.name == "group"), 1)
            out.append((n.name, group > 1))
    return out
