"""Static INT8/FP8 analysis of an FP32 model: parameters only, no image and no forward pass.

What a network's weights and batch-normalisation (BN) buffers reveal before anything is executed:

  weights     exact SQNR of every convolution / linear weight under symmetric INT8 with one scale per output
              channel (deployment setting) and per tensor, and under FP8 E4M3 with per-channel max scaling
  activations every BN stores the running mean and variance of its input, so its output channel c is modelled as
              N(beta_c, gamma_c^2 v_c / (v_c + eps)), passed through the following activation function
              (ReLU, SiLU, ...); for the per-tensor quantizer of this tensor the script reports the channel
              imbalance (90th / 10th percentile of the channel RMS), the MSE-optimal normalised range kappa
              (Eq. (5)) and the predicted injected SQNR of INT8 (Lemma 1) and FP8
  screening   SQNR_add of all modelled activation quantizers with unit propagation factors (Eq. (16)), i.e. the
              FP32-statistics screening of Section 3.8 without calibration data
  BN gain     closed form of Eq. (A3) for batch normalisations that cannot be folded into a preceding
              convolution (e.g. the pre-activation BN of DenseNet), > 1 means amplification
  structure   depthwise convolutions, sigmoid gates (squeeze-and-excitation, SiLU), concatenations implied by
              non-foldable BN, attention blocks

The activation model is an approximation (Gaussian channels, BN statistics of the training data, no residual
additions, no calibration-set clipping); Section 3.8 / Table tab:static of the article compare it with the
measured quantizer statistics.

  python scripts/43_static_analysis.py --model efficientnet_b0
  python scripts/43_static_analysis.py --model all
  python scripts/43_static_analysis.py --torchvision resnet50      # any torchvision classification model
  python scripts/43_static_analysis.py --module my_model.pt         # any model saved with torch.save(model, path)
"""
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pepai.config import load_config, results_dir

ACTIVATIONS = {nn.ReLU: "relu", nn.ReLU6: "relu6", nn.SiLU: "silu", nn.Hardswish: "hardswish", nn.GELU: "gelu",
               nn.Sigmoid: "sigmoid", nn.LeakyReLU: "leaky_relu", nn.Identity: "identity"}
INT8_LEVELS = 127
E4M3_MAX = 448.0
N_SAMPLES = 2048          # samples per channel of the Gaussian activation model (synthetic, not data)


# ----------------------------------------------------------------------------------------------- quantizers

def int8_quant(x, scale):
    return np.clip(np.round(x / scale), -INT8_LEVELS, INT8_LEVELS) * scale


def e4m3_quant(x, scale):
    """Round to the nearest E4M3 number (3 explicit mantissa bits, min normal 2^-6, max 448) after scaling."""
    v = x / scale
    a = np.minimum(np.abs(v), E4M3_MAX)
    e = np.floor(np.log2(np.maximum(a, 2.0 ** -6)))
    step = 2.0 ** (e - 3)
    return np.sign(v) * np.minimum(np.round(a / step) * step, E4M3_MAX) * scale


def sqnr_db(x, q):
    err = np.sum((q - x) ** 2)
    return float("inf") if err == 0 else float(10 * np.log10(np.sum(x ** 2) / err))


def weight_sqnr(w):
    """Exact INT8 (per channel, per tensor) and FP8 (per channel) SQNR of a weight tensor [out, ...]."""
    w = w.reshape(w.shape[0], -1).astype(np.float64)
    amax = np.maximum(np.abs(w).max(axis=1, keepdims=True), 1e-12)
    per_ch = int8_quant(w, amax / INT8_LEVELS)
    per_t = int8_quant(w, amax.max() / INT8_LEVELS)
    fp8 = e4m3_quant(w, amax / E4M3_MAX)
    return sqnr_db(w, per_ch), sqnr_db(w, per_t), sqnr_db(w, fp8)


# ------------------------------------------------------------------------------------ activation model

def act_fn(name):
    return {"relu": lambda z: np.maximum(z, 0), "relu6": lambda z: np.clip(z, 0, 6),
            "silu": lambda z: z / (1 + np.exp(-z)), "hardswish": lambda z: z * np.clip(z + 3, 0, 6) / 6,
            "gelu": lambda z: 0.5 * z * (1 + np.tanh(0.7978845608 * (z + 0.044715 * z ** 3))),
            "sigmoid": lambda z: 1 / (1 + np.exp(-z)), "leaky_relu": lambda z: np.where(z > 0, z, 0.01 * z),
            "identity": lambda z: z}[name]


def bn_output_moments(bn):
    """Per-channel mean and standard deviation of the BN output under its running statistics."""
    g = bn.weight.detach().double().numpy()
    b = bn.bias.detach().double().numpy()
    v = bn.running_var.detach().double().numpy()
    eps = getattr(bn, "eps", 1e-5)
    return b, np.abs(g) * np.sqrt(v / (v + eps))


def activation_quantizer(bn, act, rng):
    """Static model of the per-tensor quantizer of a BN(+activation) output tensor."""
    mean, std = bn_output_moments(bn)
    z = mean[:, None] + std[:, None] * rng.standard_normal((len(mean), N_SAMPLES))
    x = act_fn(act)(z)
    ch_rms = np.sqrt(np.mean(x ** 2, axis=1))
    rms = float(np.sqrt(np.mean(x ** 2)))
    if rms == 0:
        return None
    flat = x.ravel()
    amax = float(np.abs(flat).max())
    # MSE-optimal clipping range of Eq. (5) on the modelled distribution (analytical range setting)
    cands = amax * np.geomspace(0.05, 1.0, 60)
    errs = [np.mean((int8_quant(flat, a / INT8_LEVELS) - flat) ** 2) for a in cands]
    alpha = float(cands[int(np.argmin(errs))])
    q8 = int8_quant(flat, alpha / INT8_LEVELS)
    qf8 = e4m3_quant(flat, amax / E4M3_MAX)
    p10, p90 = np.percentile(ch_rms[ch_rms > 0], [10, 90]) if np.any(ch_rms > 0) else (np.nan, np.nan)
    return {"channels": len(mean), "channel_spread": float(p90 / p10) if p10 > 0 else np.inf,
            "dead_channels": int(np.sum(ch_rms < 1e-6 * max(rms, 1e-12))),
            "kappa": alpha / rms, "sqnr_int8_pred_db": 10 * np.log10(12 * INT8_LEVELS ** 2) - 20 * np.log10(alpha / rms),
            "sqnr_int8_db": sqnr_db(flat, q8), "sqnr_fp8_db": sqnr_db(flat, qf8),
            "clip_share": float(np.mean(np.abs(flat) > alpha))}


def bn_gain(bn):
    """Eq. (A3): propagation gain of a non-folded BN for equal noise power per input channel."""
    g = bn.weight.detach().double().numpy()
    b = bn.bias.detach().double().numpy()
    mu = bn.running_mean.detach().double().numpy()
    v = bn.running_var.detach().double().numpy()
    eps = getattr(bn, "eps", 1e-5)
    g2 = np.mean(g ** 2 / (v + eps)) * np.sum(mu ** 2 + v) / np.sum(g ** 2 * v / (v + eps) + b ** 2)
    return float(np.sqrt(g2))


# ------------------------------------------------------------------------------------------ traversal

def is_bn(m):
    return isinstance(m, nn.modules.batchnorm._BatchNorm) or type(m).__name__ == "FrozenBatchNorm2d"


def leaf_modules(model):
    return [(n, m) for n, m in model.named_modules() if len(list(m.children())) == 0]


FUNCTIONAL_ACTS = {"relu": "relu", "relu_": "relu", "relu6": "relu6", "silu": "silu", "hardswish": "hardswish",
                   "gelu": "gelu", "sigmoid": "sigmoid", "leaky_relu": "leaky_relu"}
_GRAPH_CACHE = {}


def graph_activations(model):
    """BN name -> activation applied to its output, read from the symbolic graph (torch.fx; nothing is executed).
    A BN whose output feeds an addition or several consumers is 'identity'. None if the model cannot be traced."""
    if id(model) in _GRAPH_CACHE:
        return _GRAPH_CACHE[id(model)]
    out = None
    try:
        import operator
        import torch.fx
        gm = torch.fx.symbolic_trace(model)
        mods = dict(gm.named_modules())
        out = {}
        for node in gm.graph.nodes:
            if node.op != "call_module" or not is_bn(mods.get(node.target)):
                continue
            users = list(node.users)
            act = "identity"
            if len(users) == 1:
                u = users[0]
                if u.op == "call_module":
                    for cls, a in ACTIVATIONS.items():
                        if isinstance(mods.get(u.target), cls):
                            act = a
                elif u.op in ("call_function", "call_method"):
                    fname = getattr(u.target, "__name__", str(u.target))
                    act = FUNCTIONAL_ACTS.get(fname, "identity")
            out[node.target] = act
    except Exception:                    # dynamic control flow etc.: fall back to the module order
        out = None
    _GRAPH_CACHE[id(model)] = out
    return out


def following_activation(leaves, i, name, model):
    """Activation applied to the output of the BN at leaves[i]: from the symbolic graph if the model can be traced,
    otherwise the next leaf module (or the parent's shared ReLU in torchvision ResNet blocks)."""
    g = graph_activations(model)
    if g is not None and name in g:
        return g[name]
    parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
    kind = type(parent).__name__
    if kind in ("Bottleneck", "BasicBlock") or name.endswith("downsample.1"):
        # torchvision ResNet: the shared ReLU follows bn1 (and bn2 of a Bottleneck); the last BN of the block and
        # the downsample BN are added to the shortcut before the ReLU, so their own output is not rectified
        last = "bn3" if kind == "Bottleneck" else "bn2"
        return "identity" if name.endswith(last) or name.endswith("downsample.1") else "relu"
    if i + 1 < len(leaves):
        nxt = leaves[i + 1][1]
        for cls, act in ACTIVATIONS.items():
            if isinstance(nxt, cls):
                return act
    return "identity"


def foldable(leaves, i):
    """A BN is folded into the convolution that directly precedes it with the same number of channels."""
    if i == 0:
        return False
    prev = leaves[i - 1][1]
    bn = leaves[i][1]
    return isinstance(prev, (nn.Conv2d, nn.Conv1d)) and prev.out_channels == bn.weight.numel()


def analyse(model, rng):
    leaves = leaf_modules(model)
    layers, acts, bns = [], [], []
    for i, (name, m) in enumerate(leaves):
        if isinstance(m, (nn.Conv2d, nn.Linear)) and m.weight is not None:
            w = m.weight.detach().double().numpy()
            if i + 1 < len(leaves) and is_bn(leaves[i + 1][1]) and foldable(leaves, i + 1):
                bn = leaves[i + 1][1]   # deployed weights: BN folded into the convolution
                scale = bn.weight.detach().double().numpy() / np.sqrt(bn.running_var.detach().double().numpy()
                                                                     + getattr(bn, "eps", 1e-5))
                w = w * scale.reshape(-1, *([1] * (w.ndim - 1)))
            pc, pt, f8 = weight_sqnr(w)
            dw = isinstance(m, nn.Conv2d) and m.groups > 1 and m.groups == m.in_channels
            layers.append({"layer": name, "type": "depthwise" if dw else type(m).__name__.lower(),
                           "params": m.weight.numel(), "w_int8_per_channel_db": pc, "w_int8_per_tensor_db": pt,
                           "w_fp8_db": f8})
        if is_bn(m):
            act = following_activation(leaves, i, name, model)
            fold = foldable(leaves, i)
            q = activation_quantizer(m, act, rng)
            if q is not None:
                acts.append({"tensor": name, "activation": act, **q})
            if not fold:
                bns.append({"bn": name, "gain_a3": bn_gain(m)})
    # tensors the BN-based model cannot see: conv/linear outputs not followed by a BN (except the final layer)
    # and other normalisations; their noise is missing from SQNR_add, which is then optimistic
    unmodelled = [n for j, (n, m) in enumerate(leaves[:-1]) if isinstance(m, (nn.Conv2d, nn.Linear))
                  and not is_bn(leaves[j + 1][1])]
    other_norms = [n for n, m in leaves if isinstance(m, (nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm2d))]
    n_sigmoid = sum(isinstance(m, (nn.Sigmoid, nn.SiLU, nn.Hardsigmoid)) for _, m in leaves)
    n_attn = sum(type(m).__name__ in ("Attention", "MultiheadAttention", "PSA") for m in model.modules())
    return pd.DataFrame(layers), pd.DataFrame(acts), pd.DataFrame(bns), {
        "sigmoid_gates": n_sigmoid, "attention_blocks": n_attn, "unmodelled_outputs": len(unmodelled),
        "unmodelled_examples": ", ".join(unmodelled[:3]), "other_norm_layers": len(other_norms),
        "other_norm_examples": ", ".join(other_norms[:3]), "graph_traced": graph_activations(model) is not None}


def summarise(name, layers, acts, bns, extra):
    s = {"model": name, "params_m": layers.params.sum() / 1e6, "conv_linear_layers": len(layers),
         "depthwise_convs": int((layers.type == "depthwise").sum()),
         "w_int8_per_channel_median_db": layers.w_int8_per_channel_db.median(),
         "w_int8_per_channel_min_db": layers.w_int8_per_channel_db.min(),
         "w_int8_per_tensor_median_db": layers.w_int8_per_tensor_db.median(),
         "w_int8_per_tensor_min_db": layers.w_int8_per_tensor_db.min(),
         "w_fp8_median_db": layers.w_fp8_db.median(),
         "act_quantizers": len(acts), "act_channel_spread_median": acts.channel_spread.median(),
         "act_channel_spread_max": acts.channel_spread.replace(np.inf, np.nan).max(),
         "act_kappa_median": acts.kappa.median(), "act_int8_sqnr_median_db": acts.sqnr_int8_db.median(),
         "act_int8_sqnr_min_db": acts.sqnr_int8_db.min(), "act_fp8_sqnr_median_db": acts.sqnr_fp8_db.median(),
         "static_sqnr_add_int8_db": -10 * np.log10(np.sum(10 ** (-acts.sqnr_int8_db / 10))),
         "static_sqnr_add_fp8_db": -10 * np.log10(np.sum(10 ** (-acts.sqnr_fp8_db / 10))),
         "nonfoldable_bn": len(bns), "bn_gain_median": bns.gain_a3.median() if len(bns) else np.nan,
         "bn_gain_max": bns.gain_a3.max() if len(bns) else np.nan, **extra}
    return s


def pct(db):
    """Relative error r = 10^(-SQNR/20) in percent of the signal, e.g. '38.0 dB (1.3 %)'."""
    return f"{db:.1f} dB ({100 * 10 ** (-db / 20):.1f} %)"


def report(s, layers, acts, bns):
    print(f"\n=== {s['model']}: {s['params_m']:.1f} M parameters, {s['conv_linear_layers']} conv/linear layers "
          f"({s['depthwise_convs']} depthwise)")
    print("  (dB values are SQNR; in brackets the relative error r = 10^(-SQNR/20) in percent of the signal)")
    print(f"  weights      INT8 per channel: median {pct(s['w_int8_per_channel_median_db'])}, "
          f"worst {pct(s['w_int8_per_channel_min_db'])}")
    print(f"               INT8 per tensor:  median {pct(s['w_int8_per_tensor_median_db'])}, "
          f"worst {pct(s['w_int8_per_tensor_min_db'])}; FP8: median {pct(s['w_fp8_median_db'])}")
    print(f"  activations  {s['act_quantizers']} BN-modelled quantizers; channel spread median "
          f"{s['act_channel_spread_median']:.1f}x (max {s['act_channel_spread_max']:.0f}x); MSE-optimal kappa median "
          f"{s['act_kappa_median']:.1f}")
    print(f"               noise injected per quantizer: INT8 median {pct(s['act_int8_sqnr_median_db'])}, "
          f"worst {pct(s['act_int8_sqnr_min_db'])}; FP8 {pct(s['act_fp8_sqnr_median_db'])}")
    print(f"  screening    all activation quantizers together (unit propagation factors, no data): "
          f"INT8 {pct(s['static_sqnr_add_int8_db'])}, FP8 {pct(s['static_sqnr_add_fp8_db'])}")
    if s["nonfoldable_bn"]:
        print(f"  BN gain      {s['nonfoldable_bn']} non-foldable BN, Eq. (A3) median {s['bn_gain_median']:.2f} "
              f"(max {s['bn_gain_max']:.2f}); > 1 amplifies upstream noise -> candidate for FP16 placement")
    print(f"  structure    {s['sigmoid_gates']} sigmoid-type activations (SiLU / gates), "
          f"{s['attention_blocks']} attention blocks")
    worst = acts.nsmallest(3, "sqnr_int8_db")
    for _, r in worst.iterrows():
        print(f"  weakest      {r.tensor} ({r.activation}): INT8 {pct(r.sqnr_int8_db)}, spread "
              f"{r.channel_spread:.0f}x, kappa {r.kappa:.1f}")
    flags = []
    if s["depthwise_convs"]:
        flags.append("depthwise convolutions with per-tensor activation scales")
    if s["act_channel_spread_max"] > 20:
        flags.append(f"strong channel imbalance (up to {s['act_channel_spread_max']:.0f}x) under one per-tensor scale")
    if s["nonfoldable_bn"] and s["bn_gain_median"] > 1:
        flags.append("non-foldable BN amplifies quantization noise")
    if s["attention_blocks"]:
        flags.append("attention block (wide activation range, clipping under distribution shift)")
    print("  flags        " + ("; ".join(flags) if flags else "none"))
    cover = [f"activations taken from the {'traced graph' if s.get('graph_traced') else 'module order (graph not traceable)'}"]
    if s.get("unmodelled_outputs"):
        cover.append(f"{s['unmodelled_outputs']} conv/linear outputs without BN are not modelled "
                     f"(e.g. {s['unmodelled_examples']})")
    if s.get("other_norm_layers"):
        cover.append(f"{s['other_norm_layers']} GroupNorm/LayerNorm layers are not modelled "
                     f"(e.g. {s['other_norm_examples']})")
    if len(cover) > 1:
        cover.append("their noise is missing, so SQNR_add is optimistic")
    print("  coverage     " + "; ".join(cover))


def scenarios(s, layers, acts):
    """Static SQNR_add (activations + weights) per deployment scenario and an order-of-magnitude INT8 loss estimate
    calibrated on the five networks of the article (static SQNR_add of the activations vs measured TensorRT loss)."""
    def add(*arrays):
        return float(-10 * np.log10(sum(np.sum(10 ** (-np.asarray(a) / 10)) for a in arrays)))
    rows = [("INT8, per-channel weights", add(acts.sqnr_int8_db, layers.w_int8_per_channel_db)),
            ("INT8, per-tensor weights", add(acts.sqnr_int8_db, layers.w_int8_per_tensor_db)),
            ("FP8 E4M3", add(acts.sqnr_fp8_db, layers.w_fp8_db)),
            ("FP16", add(np.full(len(acts), 73.66), np.full(len(layers), 73.66)))]
    refs = []
    from pepai.config import CODE_ROOT
    ref = CODE_ROOT / "reference_results" / "tables"
    if (ref / "prediction.csv").exists() and (ref / "static_analysis.csv").exists():
        pred = pd.read_csv(ref / "prediction.csv").set_index("model")
        st = pd.read_csv(ref / "static_analysis.csv").set_index("model")
        for m in [m for m in pred.index if m in st.index and m != s["model"]]:
            refs.append((m, float(st.loc[m, "static_sqnr_add_int8_db"]), float(100 * pred.loc[m, "int8_rel_loss"]),
                         float(st.loc[m, "scenario_fp8_db"]), float(100 * pred.loc[m, "fp8_rel_loss"])))
        refs.sort(key=lambda r: -r[1])
    return rows, refs


def report_scenarios(s, layers, acts):
    rows, refs = scenarios(s, layers, acts)
    print("  scenarios    activations and weights together, with unit propagation factors: the expected deviation at")
    print("               the head input if the network neither attenuates nor amplifies the noise (not the accuracy loss):")
    for label, v in rows:
        print(f"                 {label:28s} {v:6.1f} dB   ({100 * 10 ** (-v / 20):4.1f} % of the signal)")
    if refs:
        v = s["static_sqnr_add_int8_db"]
        print(f"  references   static INT8 SQNR_add of this network: {v:.1f} dB. Measured networks of the article "
              f"(static SQNR_add, measured relative INT8 loss with TensorRT):")
        placed = False
        for m, x, loss, _, _ in refs:
            if not placed and v >= x:
                print(f"                 --> {s['model']:24s} {v:5.1f} dB")
                placed = True
            print(f"                     {m:24s} {x:5.1f} dB   {loss:5.1f}%")
        if not placed:
            print(f"                 --> {s['model']:24s} {v:5.1f} dB")
        v8 = dict(rows)["FP8 E4M3"]
        print(f"               FP8 (static SQNR_add with weights {v8:.1f} dB; measured relative FP8 loss with TensorRT):")
        placed = False
        for m, _, _, x8, loss8 in sorted(refs, key=lambda r: -r[3]):
            if not placed and v8 >= x8:
                print(f"                 --> {s['model']:24s} {v8:5.1f} dB")
                placed = True
            print(f"                     {m:24s} {x8:5.1f} dB   {loss8:5.1f}%")
        if not placed:
            print(f"                 --> {s['model']:24s} {v8:5.1f} dB")
        print("               All five stayed within about 1 % in FP8, unrelated to their static FP8 SQNR_add; a network that "
              "amplifies")
        print("               the noise can still lose more (MobileNetV2: 10 % in the simulated FP8 check, Gamma_bar 3.3).")
        print("               The static analysis ranks networks (Spearman 0.9 on the five) but does not see how a "
              "network propagates the noise;")
        print("               it gives no accuracy number (no % loss for INT8 or FP8). For the relative loss: one measurement "
              "of the quantized head input")
        print("               (42_predict_int8.py, within about x2) or a direct check (44_validate_static.py).")
    return rows, refs


def load(name, cfg):
    from pepai import models
    if name in ("efficientnet_b0", "densenet121"):
        return models.load_classifier(name)
    if name == "frcnn_r50_fpn":                      # backbone + FPN (the quantized part), BN not yet folded
        import torchvision
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
        return torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1).backbone
    if name.startswith("yolov10"):
        return models.load_yolo(name, cfg["paths"]["data"] / "weights").model
    raise SystemExit(f"unknown model {name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", help="one of the article's models, or 'all'")
    g.add_argument("--torchvision", help="any torchvision classification model with default weights")
    g.add_argument("--module", help="a whole PyTorch model saved with torch.save(model, path); only parameters and "
                                    "buffers are read, nothing is executed")
    args = ap.parse_args()
    cfg = load_config()
    torch.set_grad_enabled(False)
    rng = np.random.default_rng(cfg["seed"])
    if args.module:
        from pathlib import Path
        model = torch.load(args.module, map_location="cpu", weights_only=False)
        if hasattr(model, "model") and isinstance(getattr(model, "model"), nn.Module):
            model = model.model          # e.g. an Ultralytics checkpoint wrapper
        targets = [(Path(args.module).stem, model.float().eval())]
    elif args.torchvision:
        import torchvision
        targets = [(args.torchvision, torchvision.models.get_model(args.torchvision, weights="DEFAULT").eval())]
    else:
        names = cfg["models"] if args.model == "all" else [args.model]
        targets = [(n, load(n, cfg).eval()) for n in names]
    out = results_dir(cfg, "tables", "static")
    rows = []
    for name, model in targets:
        layers, acts, bns, extra = analyse(model, rng)
        s = summarise(name, layers, acts, bns, extra)
        report(s, layers, acts, bns)
        sc, _ = report_scenarios(s, layers, acts)
        s.update({f"scenario_{lab.split(',')[0].split()[0].lower()}{'_pt' if 'per-tensor' in lab else ''}_db": v
                  for lab, v in sc})
        layers.to_csv(out / f"{name}_weights.csv", index=False)
        acts.to_csv(out / f"{name}_activations.csv", index=False)
        bns.to_csv(out / f"{name}_bn_gain.csv", index=False)
        rows.append(s)
    summary_p = results_dir(cfg, "tables") / "static_analysis.csv"
    old = pd.read_csv(summary_p) if summary_p.exists() else pd.DataFrame()
    new = pd.DataFrame(rows)
    if len(old):
        old = old[~old.model.isin(new.model)]
    pd.concat([old, new]).to_csv(summary_p, index=False)
    print(f"\nwritten: {summary_p} and {out}/<model>_*.csv")


if __name__ == "__main__":
    main()
