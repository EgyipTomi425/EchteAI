"""Noise injected by every activation quantizer, predicted from FP32 activations and the calibrated
scales alone (CPU only, ONNX Runtime; no quantized model is executed).

For every activation QuantizeLinear of the INT8 graph with scale s (range alpha = 127 s), the FP32
tensor x feeding it is collected on the first N analysis images and summarised by
  kappa          alpha / RMS(x), the normalised range of Eq. (sqnr-int8)
  sqnr_pred_db   52.87 - 20 log10(kappa), the granular-noise prediction
  sqnr_inj_db    10 log10(sum x^2 / sum (Q_s(x) - x)^2), the exact SQNR of the quantizer on these data
  clip_share     fraction of |x| > alpha
  channel_spread ratio of the 90th to the 10th percentile of the per-channel RMS (channel imbalance, which
                 a per-tensor scale cannot follow)
and linked to the convolutions that consume it, so that it can be compared with the measured
sensitivity s_n of those convolutions.
"""
import argparse

import numpy as np
import onnx
import onnxruntime as ort
import torch
import pandas as pd
from onnx import numpy_helper
from scipy.stats import spearmanr

from pepai import data, models
from pepai.config import load_config, results_dir
from pepai.models import SPECS
from pepai.sensitivity import conv_nodes, conv_sensitivity

ORDER = ["efficientnet_b0", "densenet121", "yolov10s", "yolov10x", "frcnn_r50_fpn"]


def activation_quantizers(int8_onnx):
    """{tensor: (scale, [consumer node names through Q -> DQ])} for activation quantizers."""
    g = onnx.load(str(int8_onnx)).graph
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    consumers = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    out = {}
    for n in g.node:
        if n.op_type != "QuantizeLinear" or n.input[0] in init:
            continue
        if "QuantizeLinear" in n.input[0] or "DequantizeLinear" in n.input[0]:
            continue
        scale = init.get(n.input[1])
        if scale is None or scale.size != 1 or n.output[0] not in consumers:
            continue
        users = [c.name for dq in consumers[n.output[0]] if dq.op_type == "DequantizeLinear"
                 for c in consumers.get(dq.output[0], [])]
        out[n.input[0]] = (float(scale), users)
    return out


def inputs(cfg, name, n, condition=None, severity=None):
    """Preprocessed analysis inputs on the CPU (same images as the activation analysis); detectors can
    read the corrupted copies of the same images (08_robustness.py)."""
    if SPECS[name].task == "cls":
        _, items = data.imagenetv2_split(cfg)
        return [models.classifier_preprocess(data.load_rgb(p)).unsqueeze(0).numpy() for p, _ in items[:n]]
    coco, ids = data.coco_val_ids(cfg, n)
    if name == "frcnn_r50_fpn":
        frcnn = models.load_frcnn()
        return [models.frcnn_preprocess(frcnn, [data.load_rgb(data.coco_val_path(cfg, coco, i))], device="cpu").tensors.numpy()
                for i in ids]
    path = ((lambda i: cfg["coco"]["root"] / "coco_c" / condition / str(severity) / f"{i}.png") if condition
            else (lambda i: data.coco_val_path(cfg, coco, i)))
    return [models.letterbox(data.load_rgb(path(i)))[0].unsqueeze(0).numpy() for i in ids]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=ORDER)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--condition", help="corrupted inputs (detectors), e.g. contrast")
    ap.add_argument("--severity", type=int)
    ap.add_argument("--format", choices=["int8", "fp8"], default="int8",
                    help="int8: scales of the INT8 graph; fp8: scales of the FP8 (E4M3) graph")
    args = ap.parse_args()
    cfg = load_config()
    onnx_dir, act_dir = results_dir(cfg, "onnx"), results_dir(cfg, "activations")
    sfx = ("" if args.format == "int8" else "_fp8") + (f"_{args.condition}{args.severity}" if args.condition else "")
    qmax = 127 if args.format == "int8" else 448        # largest representable multiple of the scale
    out_p = results_dir(cfg, "tables") / f"quantizer_snr{sfx}.csv"
    sum_p = results_dir(cfg, "tables") / f"quantizer_snr{sfx}_summary.csv"
    rows = [r for r in (pd.read_csv(out_p).to_dict("records") if out_p.exists() else []) if r["model"] not in args.models]
    summary = [r for r in (pd.read_csv(sum_p).to_dict("records") if sum_p.exists() else [])
               if r["model"] not in args.models]
    for name in args.models:
        quantizers = activation_quantizers(onnx_dir / f"{name}_{'int8fp32' if args.format == 'int8' else 'fp8'}.onnx")
        model = onnx.load(str(onnx_dir / f"{name}_fp32.onnx"))
        graph_inputs = {i.name for i in model.graph.input}
        produced = {o for node in model.graph.node for o in node.output}
        names = [t for t in quantizers if t in produced or t in graph_inputs]
        fetched = [t for t in names if t not in graph_inputs]
        existing = {o.name for o in model.graph.output}
        model.graph.output.extend([onnx.helper.make_empty_tensor_value_info(t) for t in fetched if t not in existing])
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 32
        sess = ort.InferenceSession(model.SerializeToString(), opts, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        acc = {t: {"x2": 0.0, "e2": 0.0, "n": 0, "clip": 0, "ch": [], "dz_n": 0, "dz_x2": 0.0} for t in names}
        for x in inputs(cfg, name, args.n, args.condition, args.severity):
            outs = dict(zip(fetched, sess.run(fetched, {input_name: x.astype(np.float32)})))
            outs.update({t: x for t in names if t in graph_inputs})
            for t, v in outs.items():
                v = v.astype(np.float64)
                s = quantizers[t][0]
                if args.format == "int8":
                    q = np.clip(np.round(v / s), -127, 127) * s
                else:                   # saturating E4M3 rounding, as QuantizeLinear(saturate=1)
                    q = torch.from_numpy(np.clip(v / s, -448, 448)).to(torch.float8_e4m3fn).double().numpy() * s
                a = acc[t]
                a["x2"] += float(np.sum(v ** 2))
                a["e2"] += float(np.sum((q - v) ** 2))
                a["n"] += v.size
                a["clip"] += int(np.sum(np.abs(v) > qmax * s))
                dead = (np.abs(v) < s / 2) & (v != 0)            # non-zero values rounded to zero
                a["dz_n"] += int(dead.sum())
                a["dz_x2"] += float(np.sum(v[dead] ** 2))
                if v.ndim == 4:
                    a["ch"].append(np.sqrt(np.mean(v ** 2, axis=(0, 2, 3))))
        for t in names:
            a = acc[t]
            s, users = quantizers[t]
            rms = np.sqrt(a["x2"] / a["n"])
            ch = np.mean(a["ch"], axis=0) if a["ch"] else None
            kappa = qmax * s / max(rms, 1e-12)
            # Channel-normalised prediction: after a depthwise convolution with folded batch normalisation the
            # output channels carry comparable power, so the SQNR is set by the mean of (alpha / sigma_c)^2.
            sqnr_ch = (10 * np.log10(12 * 127 ** 2) - 10 * np.log10(np.mean((127 * s / np.maximum(ch, 1e-12)) ** 2))
                       if ch is not None else np.nan)
            rows.append({"model": name, "tensor": t, "consumers": ";".join(users), "scale": s, "rms": rms,
                         "sqnr_channel_pred_db": sqnr_ch,
                         "kappa": kappa, "sqnr_pred_db": (10 * np.log10(12 * 127 ** 2) - 20 * np.log10(kappa)
                                                          if args.format == "int8" else -10 * np.log10(0.180 * 2.0 ** -8)),
                         "sqnr_inj_db": 10 * np.log10(a["x2"] / max(a["e2"], 1e-30)),
                         "clip_share": a["clip"] / a["n"],
                         "deadzone_share": a["dz_n"] / a["n"], "deadzone_energy_share": a["dz_x2"] / max(a["x2"], 1e-30),
                         "channels_below_step": float(np.mean(ch < s)) if ch is not None else np.nan,
                         "channel_spread": (np.percentile(ch, 90) / max(np.percentile(ch, 10), 1e-12))
                         if ch is not None and len(ch) > 1 else np.nan})
        # Only quantizers upstream of the head-input tensors can contribute to the head-input deviation.
        import json
        heads = [t["name"] for t in json.loads((act_dir / f"{name}_int8fp32_tensors.json").read_text()) if t["final"]]
        producer = {o: node for node in model.graph.node for o in node.output}
        upstream, stack = set(), list(heads)
        while stack:
            t = stack.pop()
            if t in upstream:
                continue
            upstream.add(t)
            if t in producer:
                stack.extend(producer[t].input)
        for r in rows:
            if r["model"] == name:
                r["upstream_of_head"] = r["tensor"] in upstream
        df = pd.DataFrame([r for r in rows if r["model"] == name]).reset_index(drop=True)
        # Link to the measured sensitivity of the consuming convolutions (activation analysis, 500 images).
        csv, js = act_dir / f"{name}_int8fp32.csv.gz", act_dir / f"{name}_int8fp32_tensors.json"
        if csv.exists():
            rank = conv_sensitivity(csv, js, onnx_dir / f"{name}_fp32.onnx", onnx_dir / f"{name}_int8fp32.onnx")
            sens = dict(zip(rank.node, rank.drop_db))
            sqnr_out = dict(zip(rank.node, rank.sqnr_out_db))
            dw = {n: d for n, d in conv_nodes(onnx_dir / f"{name}_fp32.onnx")}
            df["sensitivity_db"] = [max((sens[u] for u in c.split(";") if u in sens), default=np.nan)
                                    for c in df.consumers]
            df["consumer_sqnr_out_db"] = [min((sqnr_out[u] for u in c.split(";") if u in sqnr_out), default=np.nan)
                                          for c in df.consumers]
            df["depthwise"] = [any(dw.get(u, False) for u in c.split(";")) for c in df.consumers]
            for idx, r in df.iterrows():      # keep the per-tensor rows in sync with the linked columns
                rows[len(rows) - len(df) + idx].update({k: r[k] for k in ("sensitivity_db", "consumer_sqnr_out_db",
                                                                         "depthwise")})
            ok = df.dropna(subset=["sensitivity_db"])
            if len(ok) > 5:
                rho = spearmanr(-ok.sqnr_inj_db, ok.sensitivity_db)
                summary.append({"model": name, "n_quantizers": len(df), "n_linked": len(ok),
                                # Proposition 3 with all propagation factors set to one
                                "additive_gamma1_sqnr_db": -10 * np.log10(np.sum(
                                    10 ** (-df.sqnr_inj_db[df.upstream_of_head] / 10))),
                                "n_upstream": int(df.upstream_of_head.sum()),
                                "spearman_rho": rho.statistic, "p": rho.pvalue,
                                "pred_vs_inj_mae_db": float(np.mean(np.abs(df.sqnr_pred_db - df.sqnr_inj_db)
                                                                    [df.clip_share < 1e-3])),
                                "median_kappa": df.kappa.median(), "median_sqnr_inj_db": df.sqnr_inj_db.median(),
                                "median_channel_spread": df.channel_spread.median(),
                                "median_deadzone_share": df.deadzone_share.median(),
                                "median_channels_below_step": df.channels_below_step.median()})
                dwn = ok[ok.depthwise].dropna(subset=["consumer_sqnr_out_db", "sqnr_channel_pred_db"])
                if len(dwn) > 3:
                    summary[-1].update({
                        "n_depthwise": len(dwn),
                        "dw_spearman_channel_pred_vs_out": spearmanr(dwn.sqnr_channel_pred_db,
                                                                     dwn.consumer_sqnr_out_db).statistic,
                        "dw_mae_channel_pred_vs_out_db": float(np.mean(np.abs(dwn.sqnr_channel_pred_db
                                                                              - dwn.consumer_sqnr_out_db))),
                        "dw_mae_tensor_pred_vs_out_db": float(np.mean(np.abs(dwn.sqnr_inj_db
                                                                             - dwn.consumer_sqnr_out_db)))})
                print(summary[-1], flush=True)
        pd.DataFrame(rows).to_csv(out_p, index=False)
        pd.DataFrame(summary).to_csv(sum_p, index=False)
