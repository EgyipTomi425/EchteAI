"""Compare the calibrated activation scales of two INT8 QDQ graphs of the same model (CPU only).

Used to diagnose the sensitivity of INT8 accuracy to the calibration sample (29_calibration_variability.py):
for every activation quantizer, the ratio of its scale in a half-set calibration to that in the full-set
calibration, together with the operator that consumes the quantized tensor. Weight quantizers (per-channel,
data-independent) are skipped.
"""
import argparse

import numpy as np
import onnx
import pandas as pd
from onnx import numpy_helper

from pepai.config import load_config, results_dir


def activation_scales(path):
    """Scale of every activation QuantizeLinear, keyed by its input tensor, with its consumers."""
    g = onnx.load(str(path), load_external_data=False).graph
    init = {i.name: i for i in g.initializer}
    consts = {n.output[0]: n.attribute[0].t for n in g.node if n.op_type == "Constant"}
    consumers = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    rows = {}
    for n in g.node:
        if n.op_type != "QuantizeLinear" or n.input[0] in init:
            continue
        s = n.input[1]
        t = init.get(s) or consts.get(s)
        if t is None:
            continue
        scale = numpy_helper.to_array(t)
        if scale.size != 1:
            continue
        users = []
        for dq in consumers.get(n.output[0], []):
            users += [f"{u.op_type}:{u.name}" for u in consumers.get(dq.output[0], [])]
        rows[n.input[0]] = {"scale": float(scale), "consumers": ";".join(users)}
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3])
    args = ap.parse_args()
    cfg = load_config()
    full = activation_scales(results_dir(cfg, "onnx") / f"{args.model}_int8fp32.onnx")
    order = {k: i for i, k in enumerate(full)}
    out = []
    for seed in args.seeds:
        p = results_dir(cfg, "calib_variability") / f"{args.model}_s{seed}_int8fp32.onnx"
        if not p.exists():
            continue
        half = activation_scales(p)
        for k, v in full.items():
            if k in half:
                out.append({"model": args.model, "seed": seed, "tensor": k, "order": order[k],
                            "consumers": v["consumers"], "scale_full": v["scale"], "scale_half": half[k]["scale"],
                            "ratio": half[k]["scale"] / v["scale"]})
    df = pd.DataFrame(out)
    df.to_csv(results_dir(cfg, "tables") / f"calibration_scales_{args.model}.csv", index=False)
    for seed, g in df.groupby("seed"):
        lr = np.abs(np.log2(g.ratio))
        print(f"seed {seed}: {len(g)} quantizers, median |log2 ratio| {lr.median():.3f}, "
              f"share changed >25% {np.mean(lr > np.log2(1.25)):.2%}")
        print(g.assign(lr=lr).sort_values("lr", ascending=False).head(8)[
            ["order", "tensor", "consumers", "scale_full", "scale_half", "ratio"]].to_string(index=False))
