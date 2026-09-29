"""Closed-form amplification of non-folded batch normalisation versus the measured ratio (CPU only).

Inference-mode BN, y_c = gamma_c (x_c - mu_c) / sqrt(v_c + eps) + beta_c, maps a deviation e_c of its
input to gamma_c e_c / sqrt(v_c + eps). For per-tensor quantization noise, which has the same power in
every channel, and input / output signal powers mu_c^2 + v_c and gamma_c^2 v_c / (v_c + eps) + beta_c^2
(the running statistics are the moments of the input), the relative-error gain is

    a_BN^2 = mean_c[gamma_c^2 / (v_c + eps)] * sum_c (mu_c^2 + v_c) / sum_c (gamma_c^2 v_c / (v_c + eps) + beta_c^2).

It exceeds one when mean removal shrinks the signal (large mu_c^2 / v_c, as after ReLU and concatenation)
and depends only on the FP32 parameters. The measured a_n is the median over the analysis images of
r_out / max r_in (as in 12_figures.fig_operator_amplification).
"""
import json

import numpy as np
import onnx
import pandas as pd
from onnx import numpy_helper
from scipy.stats import spearmanr

from pepai.activations import load_activation_table, resolved_inputs
from pepai.config import load_config, results_dir


def predicted_gain(fp32_onnx):
    g = onnx.load(str(fp32_onnx)).graph
    init = {i.name: numpy_helper.to_array(i).astype(np.float64) for i in g.initializer}
    out = {}
    for n in g.node:
        if n.op_type != "BatchNormalization":
            continue
        gam, bet, mu, var = (init[i] for i in n.input[1:5])
        eps = next((a.f for a in n.attribute if a.name == "epsilon"), 1e-5)
        num = np.mean(gam ** 2 / (var + eps)) * np.sum(mu ** 2 + var)
        den = np.sum(gam ** 2 * var / (var + eps) + bet ** 2)
        out[n.output[0]] = float(np.sqrt(num / den))
    return out


if __name__ == "__main__":
    cfg = load_config()
    name = "densenet121"
    d = results_dir(cfg, "activations")
    fp32 = results_dir(cfg, "onnx") / f"{name}_fp32.onnx"
    pred = predicted_gain(fp32)
    df = load_activation_table(d / f"{name}_int8fp32.csv.gz", usecols=["image", "tensor", "op", "rel_l2"])
    piv = df.pivot_table(index="image", columns="tensor", values="rel_l2")
    infos = json.loads((d / f"{name}_int8fp32_tensors.json").read_text())
    resolved = resolved_inputs(fp32, [t["name"] for t in infos])
    rows = []
    for t in infos:
        if t["op"] != "BatchNormalization" or t["name"] not in pred or t["name"] not in piv:
            continue
        inputs = [i for i in resolved.get(t["node"], t["inputs"]) if i in piv and i != t["name"]]
        if not inputs:
            continue
        measured = (piv[t["name"]] / piv[inputs].max(axis=1).clip(lower=1e-9)).median()
        rows.append({"model": name, "tensor": t["name"], "order": t["order"], "measured": measured,
                     "predicted": pred[t["name"]]})
    r = pd.DataFrame(rows).sort_values("order")
    r.to_csv(results_dir(cfg, "tables") / "bn_amplification.csv", index=False)
    rho = spearmanr(r.measured, r.predicted)
    print(f"{len(r)} BN nodes; median measured {r.measured.median():.3f}, predicted {r.predicted.median():.3f}; "
          f"Spearman rho {rho.statistic:.3f} (p = {rho.pvalue:.1e}); first layer "
          f"{r.measured.iloc[0]:.2f} vs {r.predicted.iloc[0]:.2f}")
