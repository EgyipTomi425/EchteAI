"""All manuscript figures from results/ (figures whose inputs are missing are skipped)."""
import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from pepai.activations import load_activation_table, resolved_inputs
from pepai.config import load_config, results_dir
from pepai.models import IMAGENET_MEAN, IMAGENET_STD
from pepai.plots import (BLUE_700, GRID as GRID_LINE, INK_2, MODEL_COLORS, MODEL_LABELS, MODEL_MARKERS, MUTED,
                         PRECISION_COLORS, PRECISION_LABELS, SEQUENTIAL, save, style)

ORDER = ["frcnn_r50_fpn", "yolov10s", "yolov10x", "efficientnet_b0", "densenet121"]


def symlog(x, lin=1e-3):
    return np.sign(x) * np.log10(1 + np.abs(x) / lin)


def symlog_ticks(lin=1e-3, decades=(-2, -1, 0, 1)):
    vals = [0.0] + [s * 10.0 ** d for d in decades for s in (1, -1)]
    return sorted(vals), [f"{v:g}" for v in sorted(vals)]


def resize_nearest(mask, shape):
    ys = (np.arange(shape[0]) * mask.shape[0] / shape[0]).astype(int)
    xs = (np.arange(shape[1]) * mask.shape[1] / shape[1]).astype(int)
    return mask[np.ix_(ys, xs)]


def fig_activation_maps(cfg, out, name="frcnn_r50_fpn", quant="int8fp32", n_images=2):
    path = results_dir(cfg, "activations") / f"{name}_{quant}_maps.npz"
    if not path.exists():
        return
    maps = np.load(path)
    keys = sorted({k.rsplit("_", 1)[0] for k in maps.files if k.endswith("_input")})[:n_images]
    fig, axes = plt.subplots(len(keys), 5, figsize=(12, 2.3 * len(keys)), squeeze=False)
    for row, key in zip(axes, keys):
        x = maps[f"{key}_input"].transpose(1, 2, 0)
        if not name.startswith("yolov10"):       # Faster R-CNN and classifiers see normalised inputs
            x = x * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
        f, q = maps[f"{key}_fp32"], maps[f"{key}_{quant}"]
        vmin, vmax = f.min(), f.max()                         # one colour scale for FP32 and INT8
        diff = np.abs(q - f)
        rel = diff / (np.abs(f) + 1e-6)
        panels = [(np.clip(x, 0, 1), None, "Input"), (f, (vmin, vmax), "FP32 activation"),
                  (q, (vmin, vmax), "INT8 activation"), (diff, None, "|INT8 - FP32|"),
                  (rel, "log", "Relative deviation")]
        for ax, (img, scale, title) in zip(row, panels):
            ax.grid(False)
            ax.set_xticks([]), ax.set_yticks([])
            if scale is None and img.ndim == 3:
                ax.imshow(img)
            elif scale == "log":
                im = ax.imshow(np.clip(img, 1e-3, None), cmap=SEQUENTIAL, norm=LogNorm(1e-3, 10))
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            else:
                im = ax.imshow(img, cmap=SEQUENTIAL, vmin=scale[0] if scale else None,
                               vmax=scale[1] if scale else None)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            if f"{key}_mask" in maps.files and img.ndim == 2:
                ax.contour(resize_nearest(maps[f"{key}_mask"], img.shape), levels=[0.5], colors=[MUTED],
                           linewidths=0.6)
            ax.set_title(title)
    fig.suptitle(f"{MODEL_LABELS[name]}: channel-max activations averaged over all convolutions", x=0.01,
                 ha="left", color=INK_2)
    save(fig, out / "R1_activation_maps")


def fig_hexbin(cfg, out, quant="int8fp32"):
    d = results_dir(cfg, "activations")
    models = [m for m in ORDER if (d / f"{m}_{quant}_pairs.npy").exists()]
    if not models:
        return
    fig, axes = plt.subplots(1, len(models), figsize=(3.1 * len(models), 3.0), squeeze=False)
    ticks, labels = symlog_ticks()
    for ax, m in zip(axes[0], models):
        p = np.load(d / f"{m}_{quant}_pairs.npy")
        hb = ax.hexbin(symlog(p[:, 0]), symlog(p[:, 1]), gridsize=60, bins="log", cmap=SEQUENTIAL,
                       mincnt=1, linewidths=0)
        ax.set_xticks(symlog(np.array(ticks)), labels, rotation=90, fontsize=6)
        ax.set_yticks(symlog(np.array(ticks)), labels, fontsize=6)
        ax.set_title(MODEL_LABELS[m])
        ax.set_xlabel("FP32 activation (symlog)")
        ax.grid(False)
    axes[0][0].set_ylabel("INT8 - FP32 (symlog)")
    fig.colorbar(hb, ax=axes[0].tolist(), label="samples (log)", fraction=0.02)
    save(fig, out / "R2_hexbin")


def stage_of(model, tensor):
    """Coarse architectural stage of a tensor, for stage boundaries in layer-wise plots."""
    if model == "frcnn_r50_fpn":
        if "fpn" in tensor:
            return "FPN"
        m = re.search(r"layer(\d)", tensor)
        return f"res{int(m.group(1)) + 1}" if m else "stem"
    if model.startswith("yolov10"):
        m = re.search(r"/model\.(\d+)/", tensor)
        i = int(m.group(1)) if m else 0
        return "backbone" if i <= 10 else ("neck" if i <= 22 else "head")
    if model == "efficientnet_b0":
        m = re.search(r"features\.(\d+)", tensor)
        return f"s{m.group(1)}" if m else "head"
    m = re.search(r"(denseblock|transition)(\d)", tensor)
    return (("B" if m.group(1) == "denseblock" else "T") + m.group(2)) if m else "stem"


def layer_table(cfg, name, quant="int8fp32"):
    path = results_dir(cfg, "activations") / f"{name}_{quant}.csv.gz"
    if not path.exists():
        return None
    df = load_activation_table(path)
    conv = df[df.op == "Conv"].copy()
    orders = sorted(conv.order.unique())
    index = {o: i for i, o in enumerate(orders)}
    conv["layer"] = conv.order.map(index)
    g = conv.groupby("layer")
    t = pd.DataFrame({
        "sqnr_med": g.sqnr_db.median(), "sqnr_q1": g.sqnr_db.quantile(0.25), "sqnr_q3": g.sqnr_db.quantile(0.75),
        "mre_med": g.mre_proj.median() * 100, "mre_q1": g.mre_proj.quantile(0.25) * 100,
        "mre_q3": g.mre_proj.quantile(0.75) * 100,
        "tensor": g.tensor.first(),
    }).reset_index()
    t["stage"] = [stage_of(name, x) for x in t.tensor]
    return t


def draw_stages(ax, t, label=True):
    """Boundary at the first convolution of every stage (stages can interleave in topological order)."""
    firsts = t.groupby("stage", sort=False).layer.min().sort_values()
    starts = firsts.tolist() + [t.layer.max() + 1]
    for k, (stage, x0) in enumerate(firsts.items()):
        if x0 > 0:
            ax.axvline(x0 - 0.5, color=GRID_LINE, linewidth=0.9, zorder=0)
        if label and (starts[k + 1] - x0) >= 0.06 * starts[-1]:     # narrow stages: boundary only
            ax.text((x0 + starts[k + 1] - 1) / 2, 1.01, stage, transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=5.5, color=MUTED)


def fig_propagation(cfg, out):
    tabs = {m: layer_table(cfg, m) for m in ORDER}
    tabs = {m: t for m, t in tabs.items() if t is not None}
    if not tabs:
        return
    fig, axes = plt.subplots(2, len(tabs), figsize=(3.3 * len(tabs), 5.2), squeeze=False)
    for j, (m, t) in enumerate(tabs.items()):
        f16 = layer_table(cfg, m, "fp16")
        f8 = layer_table(cfg, m, "fp8")
        for i, (key, label) in enumerate((("sqnr", "SQNR (dB)"), ("mre", "MRE$_{proj}$ (%)"))):
            ax = axes[i][j]
            ax.fill_between(t.layer, t[f"{key}_q1"], t[f"{key}_q3"], color=PRECISION_COLORS["int8"],
                            alpha=0.22, linewidth=0)
            ax.plot(t.layer, t[f"{key}_med"], color=PRECISION_COLORS["int8"], linewidth=1.2, label="INT8")
            if f8 is not None:
                ax.plot(f8.layer, f8[f"{key}_med"], color=PRECISION_COLORS["fp8"], linewidth=1.0, label="FP8")
            if f16 is not None:
                ax.plot(f16.layer, f16[f"{key}_med"], color=PRECISION_COLORS["fp16"], linewidth=0.9, label="FP16")
            draw_stages(ax, t, label=(i == 0))
            if key == "mre":
                ax.set_yscale("log")
            if i == 0:
                ax.set_title(MODEL_LABELS[m], pad=12)
            if j == 0:
                ax.set_ylabel(label)
            if i == 1:
                ax.set_xlabel("convolution index (forward order)")
            ax.grid(axis="x", visible=False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, fontsize=8, bbox_to_anchor=(1.0, 1.04))
    save(fig, out / "R4_propagation")


def fig_layer_profile(cfg, out):
    """Where deviation is injected: SQNR drop across every quantized convolution, by depth."""
    from pepai.sensitivity import conv_sensitivity
    d = results_dir(cfg, "activations")
    models = [m for m in ORDER if (d / f"{m}_int8fp32.csv.gz").exists()]
    if not models:
        return
    fig, axes = plt.subplots(len(models), 1, figsize=(10, 2.1 * len(models)), squeeze=False)
    fig.subplots_adjust(hspace=0.6)
    for ax, m in zip(axes[:, 0], models):
        r = conv_sensitivity(d / f"{m}_int8fp32.csv.gz", d / f"{m}_int8fp32_tensors.json",
                             results_dir(cfg, "onnx") / f"{m}_fp32.onnx",
                             results_dir(cfg, "onnx") / f"{m}_int8fp32.onnx").sort_values("order")
        t = layer_table(cfg, m)
        pos = {tensor: layer for tensor, layer in zip(t.tensor, t.layer)}
        r["layer"] = r.tensor.map(pos)
        r = r.dropna(subset=["layer"])
        ax.bar(r.layer, r.drop_db.clip(lower=0), width=0.8, color=PRECISION_COLORS["int8"])
        top = r.nlargest(3, "drop_db")
        for _, row in top.iterrows():
            ax.annotate(f"{row.drop_db:.1f}", (row.layer, row.drop_db), textcoords="offset points", xytext=(0, 2),
                        ha="center", fontsize=6, color=INK_2)
        draw_stages(ax, t)
        ax.set_ylabel(f"{MODEL_LABELS[m]}\nSQNR drop (dB)", fontsize=7)
        ax.grid(axis="x", visible=False)
    axes[-1][0].set_xlabel("convolution index (forward order)")
    save(fig, out / "S_layer_profile")


def fig_operator_amplification(cfg, out, quant="int8fp32"):
    d = results_dir(cfg, "activations")
    rows = []
    for m in ORDER:
        csv, js = d / f"{m}_{quant}.csv.gz", d / f"{m}_{quant}_tensors.json"
        if not csv.exists():
            continue
        df = load_activation_table(csv, usecols=["image", "tensor", "op", "rel_l2", "sqnr_db", "cosine", "abs_mean"])
        piv = df.pivot_table(index="image", columns="tensor", values="rel_l2")
        infos = json.loads(js.read_text())
        resolved = resolved_inputs(results_dir(cfg, "onnx") / f"{m}_fp32.onnx", [t["name"] for t in infos])
        for t in infos:
            inputs = [i for i in resolved.get(t["node"], t["inputs"]) if i in piv and i != t["name"]]
            if t["name"] not in piv or not inputs:
                continue
            ratio = piv[t["name"]] / piv[inputs].max(axis=1).clip(lower=1e-9)
            rows.append({"model": m, "op": t["op"], "ratio": ratio.median()})
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(results_dir(cfg, "tables") / "operator_amplification.csv", index=False)
    from matplotlib.ticker import FixedLocator, NullFormatter
    short = {"BatchNormalization": "BatchNorm", "GlobalAveragePool": "GlobalAvgPool", "AveragePool": "AvgPool"}
    models = [m for m in ORDER if m in set(df.model)]
    fig, axes = plt.subplots(1, len(models), figsize=(2.9 * len(models), 2.9), squeeze=False, sharex=True)
    for ax, m in zip(axes[0], models):
        g = df[df.model == m]
        ops = g.groupby("op").ratio.median().sort_values().index.tolist()
        for i, op in enumerate(ops):
            v = g[g.op == op].ratio
            ax.plot([v.quantile(0.25), v.quantile(0.75)], [i, i], color=PRECISION_COLORS["fp32"], alpha=0.35,
                    linewidth=4, solid_capstyle="round")
            ax.plot(v.median(), i, "o", color=BLUE_700, markersize=5)
            ax.annotate(f"{v.median():.2f}", (v.median(), i), textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=6, color=INK_2)
        ax.axvline(1.0, color=MUTED, linewidth=0.8, linestyle="--")
        ax.set_yticks(range(len(ops)), [f"{short.get(o, o)} ({(g.op == o).sum()})" for o in ops])
        ax.set_ylim(-0.6, len(ops) - 0.3)
        ax.set_xscale("log")
        ax.set_xlim(0.2, 6.0)
        ax.xaxis.set_major_locator(FixedLocator([0.25, 0.5, 1, 2, 4]))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(MODEL_LABELS[m])
        ax.set_xlabel("amplification $a_n$")
    fig.tight_layout(w_pad=1.2)
    save(fig, out / "M3_operator_amplification")


def fig_speed(cfg, out, bench_name="benchmark.csv"):
    """a, b: speed-up over strict FP32 at batch 1 and 8; c: deployed weight size; d: energy per image."""
    path = results_dir(cfg, "tables") / bench_name
    if not path.exists():
        return
    b = pd.read_csv(path).groupby(["model", "precision", "batch"]).median(numeric_only=True).reset_index()
    precisions = [p for p in ("fp32", "fp16", "int8", "fp8") if p in set(b.precision)]
    models = [m for m in ORDER if m in set(b.model)]
    labels = [MODEL_LABELS[m].replace(" R50-FPN", "") for m in models]
    sizes_p = results_dir(cfg, "tables") / "models.csv"
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.2))

    def bars(ax, values, title, ylabel, label_prec="int8", fmt_label=lambda v: f"{v:.1f}x", ref_line=None):
        shown = [p for p in precisions if p in values]
        w = 0.8 / len(shown)
        for i, p in enumerate(shown):
            x = np.arange(len(models)) + (i - (len(shown) - 1) / 2) * w
            h = ax.bar(x, values[p], w * 0.92, color=PRECISION_COLORS[p], label=PRECISION_LABELS[p])
            if p == label_prec:
                ax.bar_label(h, labels=[fmt_label(v) if np.isfinite(v) else "" for v in values[p]], fontsize=6,
                             color=INK_2, padding=2)
        if ref_line is not None:
            ax.axhline(ref_line, color=MUTED, linewidth=0.8)
        ax.set_xticks(np.arange(len(models)), labels, rotation=15, ha="right")
        ax.set_title(title, loc="left")
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", visible=False)

    def value(m, p, bs, col):
        v = b[(b.model == m) & (b.precision == p) & (b.batch == bs)][col]
        return v.iloc[0] if len(v) else np.nan

    for k, bs in enumerate((1, 8)):
        vals = {p: [value(m, "fp32", bs, "graph_p50_ms") / value(m, p, bs, "graph_p50_ms") for m in models]
                for p in precisions if p != "fp32"}
        bars(axes[0][k], vals, f"{'ab'[k]}  speed-up over strict FP32, batch {bs}", "FP32 p50 / p50", ref_line=1.0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=PRECISION_COLORS[p]) for p in precisions]
    fig.legend(handles, [PRECISION_LABELS[p] for p in precisions], loc="upper center", ncol=len(precisions),
               bbox_to_anchor=(0.5, 1.03), fontsize=8)
    if sizes_p.exists():
        sz = pd.read_csv(sizes_p).set_index("model")
        vals = {p: [sz.loc[m, f"weights_mb_{p}"] if m in sz.index and f"weights_mb_{p}" in sz else np.nan
                    for m in models] for p in precisions}
        bars(axes[1][0], vals, "c  deployed weight size", "MB",
             fmt_label=lambda v: f"{v:.0f}" if v >= 10 else f"{v:.1f}")
    else:
        axes[1][0].set_axis_off()
    if "energy_j_per_img" in b:
        vals = {p: [1000 * value(m, p, 8, "energy_j_per_img") for m in models] for p in precisions}
        bars(axes[1][1], vals, "d  GPU energy per image, batch 8", "mJ per image",
             fmt_label=lambda v: f"{v:.1f}" if v >= 1 else f"{v:.2f}")
        axes[1][1].set_yscale("log")
        axes[1][1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    else:
        axes[1][1].set_axis_off()
    fig.tight_layout(h_pad=1.5)
    save(fig, out / "R5_speedup")


def fig_selective(cfg, out):
    path = results_dir(cfg, "tables") / "selective.csv"
    if not path.exists():
        return
    s = pd.read_csv(path)
    if "strategy" not in s:
        s["strategy"] = "pepai"
    acc_p = results_dir(cfg, "tables") / "accuracy.csv"
    acc = pd.read_csv(acc_p).set_index(["model", "precision"]) if acc_p.exists() else None
    styles = {"pepai": ("PEP-AI ranking", "#2a78d6", "-", "o"),
              "pepai_iter": ("PEP-AI iterative", BLUE_700, "--", "s"),
              "random": ("random layers (mean, min-max)", MUTED, ":", "^"),
              "depthwise": ("all depthwise convs", INK_2, "", "D")}
    for m, g in s.groupby("model"):
        metric = "top1" if "top1" in g and g.top1.notna().any() else "mAP"
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))
        for strat, (label, color, ls, marker) in styles.items():
            h = g[g.strategy == strat]
            if h.empty:
                continue
            agg = h.groupby("k").agg(acc=(metric, "mean"), lo=(metric, "min"), hi=(metric, "max"),
                                     lat=("p50_ms_bs8", "mean")).reset_index()
            axes[0].plot(agg.k, agg.acc * 100, linestyle=ls or "none", marker=marker, color=color, label=label,
                         markersize=4, linewidth=1.5)
            if strat == "random":
                axes[0].fill_between(agg.k, agg.lo * 100, agg.hi * 100, color=color, alpha=0.15, linewidth=0)
            axes[1].plot(agg.lat, agg.acc * 100, linestyle=ls or "none", marker=marker, color=color, label=label,
                         markersize=4, linewidth=1.5)
        if acc is not None:
            for p_ in ("fp32", "fp16"):
                if (m, p_) in acc.index:
                    v = acc.loc[(m, p_), metric] * 100
                    for ax in axes:
                        ax.axhline(v, color=PRECISION_COLORS[p_], linewidth=0.9, linestyle="-.")
                    axes[0].annotate(PRECISION_LABELS[p_], (g.k.max(), v), textcoords="offset points",
                                     xytext=(-2, 3 if p_ == "fp32" else -9), ha="right", fontsize=7, color=INK_2)
        axes[0].set_xlabel("Conv layers kept in FP16 (k)")
        axes[0].set_ylabel(f"{'top-1' if metric == 'top1' else 'mAP'} (%)")
        axes[1].set_xlabel("p50 latency, batch 8 (ms)")
        axes[0].legend(fontsize=7, loc="lower right")
        save(fig, out / f"S_selective_{m}")


def surface_table(cfg, name, condition, quant="int8fp32", n=100):
    """Median MRE_proj per (relative depth, severity); severity 0 = clean images (same image ids)."""
    d = results_dir(cfg, "activations")
    clean_path = d / f"{name}_{quant}.csv.gz"
    if not clean_path.exists():
        return None
    parts = []
    for sev in range(0, 6):
        path = clean_path if sev == 0 else d / f"{name}_{quant}_{condition}{sev}.csv.gz"
        if not path.exists():
            return None
        df = pd.read_csv(path, usecols=["image", "tensor", "op", "order", "mre_proj", "sqnr_db"])
        df = df[df.op == "Conv"]
        if sev == 0:
            ids = pd.read_csv(d / f"{name}_{quant}_{condition}1.csv.gz", usecols=["image"]).image.unique()
            df = df[df.image.isin(ids)]
        parts.append(df.assign(severity=sev))
    df = pd.concat(parts)
    orders = sorted(df.order.unique())
    df["depth"] = df.order.map({o: i / (len(orders) - 1) for i, o in enumerate(orders)})
    return df.groupby(["severity", "depth"]).agg(mre=("mre_proj", "median"), sqnr=("sqnr_db", "median")).reset_index()


def fig_surface(cfg, out, name="frcnn_r50_fpn", conditions=("fog", "dark")):
    """Median MRE_proj per (relative depth, severity) as heatmaps on a shared colour scale."""
    tabs = {c: surface_table(cfg, name, c) for c in conditions}
    tabs = {c: t for c, t in tabs.items() if t is not None}
    if not tabs:
        return
    pivs = {}
    for cond, t in tabs.items():
        piv = t.pivot(index="severity", columns="depth", values="mre").sort_index() * 100
        pivs[cond] = piv.T.rolling(3, center=True, min_periods=1).median().T   # light smoothing along depth
    vmax = max(np.nanmax(v.values) for v in pivs.values())
    fig, axes = plt.subplots(1, len(pivs), figsize=(4.6 * len(pivs), 2.6), squeeze=False)
    for ax, (cond, piv) in zip(axes[0], pivs.items()):
        im = ax.imshow(piv.values, aspect="auto", origin="lower", cmap=SEQUENTIAL, vmin=0, vmax=vmax,
                       extent=(0, 1, -0.5, piv.index.max() + 0.5), interpolation="nearest")
        ax.set_yticks(piv.index.values, ["clean"] + [str(v) for v in piv.index.values[1:]])
        ax.set_xlabel("relative depth of the convolution")
        if ax is axes[0][0]:
            ax.set_ylabel("severity")
        ax.set_title(CONDITION_LABELS.get(cond, cond), loc="left")
        ax.grid(False)
    cb = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.03, pad=0.02)
    cb.set_label("median MRE$_{proj}$ (%)")
    save(fig, out / "S_severity_surface")


def fig_robustness(cfg, out):
    path = results_dir(cfg, "tables") / "robustness.csv"
    if not path.exists():
        return
    r = pd.read_csv(path)
    acc = r[r.precision.isin(["fp32", "fp16", "int8"])]
    models = [m for m in ORDER if m in set(acc.model)]
    conds = [c for c in acc.condition.unique() if c != "clean"]
    fig, axes = plt.subplots(len(models), len(conds), figsize=(1.9 * len(conds), 1.9 * len(models)),
                             squeeze=False, sharex=True, sharey="row")
    for i, m in enumerate(models):
        clean = acc[(acc.model == m) & (acc.condition == "clean")]
        for j, c in enumerate(conds):
            ax = axes[i][j]
            for p in ("fp32", "fp16", "int8"):
                g = acc[(acc.model == m) & (acc.condition == c) & (acc.precision == p)].sort_values("severity")
                base = clean[clean.precision == p].mAP
                xs = [0] + g.severity.tolist()
                ys = ([base.iloc[0]] if len(base) else [np.nan]) + g.mAP.tolist()
                ax.plot(xs, np.array(ys) * 100, "o-", color=PRECISION_COLORS[p], markersize=3, linewidth=1.3,
                        label=PRECISION_LABELS[p])
            if i == 0:
                ax.set_title(CONDITION_LABELS.get(c, c.replace("_", " ")))
            if j == 0:
                ax.set_ylabel(f"{MODEL_LABELS[m]}\nmAP (%)")
            if i == len(models) - 1:
                ax.set_xlabel("severity")
    axes[0][0].legend(loc="lower left", fontsize=6)
    fig.text(0.5, -0.01, "FP16 coincides with FP32 in every panel.", ha="center", fontsize=7, color=INK_2)
    save(fig, out / "S_robustness")


def detection_level_auc(det, target, seed):
    """Cross-validated AUC of logistic models predicting a detection flip."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    y = det[target].values
    out = {}
    if y.min() == y.max():
        return out
    det = det.assign(margin=-(det.score - 0.5).abs(), log_mre=np.log(det.local_mre + 1e-6))
    for label, cols in (("local deviation", ["log_mre"]), ("score margin", ["margin"]),
                        ("margin + size", ["margin", "log_area"]),
                        ("margin + size + deviation", ["margin", "log_area", "log_mre"])):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        p = cross_val_predict(clf, det[cols].values, y, cv=cv, method="predict_proba")[:, 1]
        out[label] = roc_auc_score(y, p)
    return out


def fig_risk(cfg, out):
    from sklearn.metrics import roc_auc_score, roc_curve
    d = results_dir(cfg, "risk")
    models = [m for m in ("frcnn_r50_fpn", "yolov10s", "yolov10x") if (d / f"{m}_int8fp32.csv").exists()]
    if not models:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5))
    rows = []
    for m in models:
        df = pd.read_csv(d / f"{m}_int8fp32.csv")
        for target in ("error", "error_tol"):
            y = df[target].values
            if y.min() == y.max():
                continue
            auc = roc_auc_score(y, df.mre_proj)
            rng = np.random.default_rng(cfg["seed"])
            boots = []
            for _ in range(2000):
                idx = rng.integers(0, len(df), len(df))
                if y[idx].min() != y[idx].max():
                    boots.append(roc_auc_score(y[idx], df.mre_proj.values[idx]))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append({"model": m, "level": "image", "target": target, "predictor": "head MRE_proj",
                         "auc": auc, "auc_lo": lo, "auc_hi": hi, "rate": y.mean(),
                         "share_mre_ge_6pct": (df.mre_proj >= 0.06).mean()})
            rows.append({"model": m, "level": "image", "target": target, "predictor": "n FP32 detections",
                         "auc": roc_auc_score(y, df.n_fp32), "rate": y.mean()})
        det_p = d / f"{m}_int8fp32_detections.csv"
        if det_p.exists():
            det = pd.read_csv(det_p)
            for target in ("flip", "flip_tol"):
                for label, auc in detection_level_auc(det, target, cfg["seed"]).items():
                    rows.append({"model": m, "level": "detection", "target": target, "predictor": label,
                                 "auc": auc, "rate": det[target].mean(), "n": len(det)})
            fpr, tpr, _ = roc_curve(det.flip_tol, np.log(det.local_mre + 1e-6))
            marker, ls = MODEL_MARKERS[m]
            auc = roc_auc_score(det.flip_tol, np.log(det.local_mre + 1e-6))
            axes[0].plot(fpr, tpr, color=MODEL_COLORS[m], linewidth=1.5, linestyle=ls,
                         label=f"{MODEL_LABELS[m]} (AUC {auc:.2f})")
            bins = np.quantile(det.local_mre, np.linspace(0, 1, 11))
            det["bin"] = pd.cut(det.local_mre, np.unique(bins), include_lowest=True)
            g = det.groupby("bin", observed=True).agg(x=("local_mre", "median"), y=("flip_tol", "mean"))
            axes[1].plot(g.x * 100, g.y * 100, marker=marker, linestyle=ls, color=MODEL_COLORS[m],
                         markersize=4, linewidth=1.5, label=MODEL_LABELS[m])
            axes[1].annotate(MODEL_LABELS[m], (g.x.iloc[-1] * 100, g.y.iloc[-1] * 100), textcoords="offset points",
                             xytext=(-4, 6), ha="right", fontsize=7, color=INK_2)
    axes[0].plot([0, 1], [0, 1], color=MUTED, linewidth=0.8, linestyle=":")
    axes[0].set_xlabel("false positive rate")
    axes[0].set_ylabel("true positive rate")
    axes[0].set_title("a  vanishing detections vs local deviation", loc="left")
    axes[0].legend(fontsize=7, loc="lower right")
    axes[1].axvline(6, color=MUTED, linewidth=0.8, linestyle="--")
    axes[1].annotate("6% (conference paper)", (6, axes[1].get_ylim()[1]), textcoords="offset points", xytext=(3, -10),
                     fontsize=7, color=INK_2)
    axes[1].set_xscale("log")
    axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    axes[1].set_xlabel("local head-input MRE$_{proj}$ (%), decile medians")
    axes[1].set_ylabel("detections vanishing under INT8 (%)")
    axes[1].set_title("b  vanishing rate per deviation decile", loc="left")
    pd.DataFrame(rows).to_csv(results_dir(cfg, "tables") / "risk_auc.csv", index=False)
    save(fig, out / "R_risk")


def fig_aibo(cfg, out):
    """a: annual fleet energy; b: CO2 saved vs FP32 (grid-intensity range); c: accuracy vs electricity cost;
    d: break-even penalty per critical error (Eq. breakeven)."""
    path = results_dir(cfg, "tables") / "aibo.csv"
    if not path.exists():
        return
    a = pd.read_csv(path)
    a = a[a.energy_basis == "energy_j_per_img"]
    models = [m for m in ORDER if m in set(a.model)]
    labels = [MODEL_LABELS[m].replace(" R50-FPN", "") for m in models]
    precisions = [p for p in ("fp32", "fp16", "int8", "fp8") if p in set(a.precision)]
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.6))
    (ax_e, ax_c), (ax_t, ax_b) = axes

    def grouped(ax, col, shown, scale=1.0, err=None):
        w = 0.8 / len(shown)
        for i, p in enumerate(shown):
            g = a[a.precision == p].set_index("model").reindex(models)
            x = np.arange(len(models)) + (i - (len(shown) - 1) / 2) * w
            ax.bar(x, g[col] * scale, w * 0.92, color=PRECISION_COLORS[p], label=PRECISION_LABELS[p])
            if err is not None:
                lo, hi = g[err[0]] * scale, g[err[1]] * scale
                ax.errorbar(x, g[col] * scale, yerr=[g[col] * scale - lo, hi - g[col] * scale], fmt="none",
                            ecolor=INK_2, elinewidth=0.8, capsize=2)
        ax.set_xticks(np.arange(len(models)), labels, rotation=15, ha="right")
        ax.grid(axis="x", visible=False)

    grouped(ax_e, "kwh_per_year", precisions, 1e-3)
    ax_e.set_yscale("log")
    ax_e.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax_e.set_ylabel("MWh per year")
    ax_e.set_title("a  fleet inference energy", loc="left")
    grouped(ax_c, "tco2_saved_ref", [p for p in precisions if p != "fp32"], err=("tco2_saved_low", "tco2_saved_high"))
    ax_c.set_ylabel("t CO$_2$ saved per year")
    ax_c.set_title("b  emissions saved vs FP32 (100-400 g/kWh)", loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=PRECISION_COLORS[p]) for p in precisions]
    fig.legend(handles, [PRECISION_LABELS[p] for p in precisions], loc="upper center", ncol=len(precisions),
               bbox_to_anchor=(0.5, 1.02), fontsize=8)

    acc_p = results_dir(cfg, "tables") / "accuracy.csv"
    if acc_p.exists():
        acc = pd.read_csv(acc_p).set_index(["model", "precision"])
        for m in models:
            pts = []
            for p in precisions:
                r = a[(a.model == m) & (a.precision == p)]
                if r.empty or (m, p) not in acc.index:
                    continue
                metric = "top1" if pd.notna(acc.loc[(m, p)].get("top1")) else "mAP"
                pts.append((r.eur_per_year_ref.iloc[0] / 1000, 100 * acc.loc[(m, p), metric], p))
            if not pts:
                continue
            pts.sort()
            xs, ys, _ = zip(*pts)
            ax_t.plot(xs, ys, color=GRID_LINE, linewidth=1, zorder=1)
            for x_, y_, p in pts:
                ax_t.scatter(x_, y_, color=PRECISION_COLORS[p], s=26, zorder=2, edgecolor="white", linewidth=0.5)
            ax_t.annotate(MODEL_LABELS[m].replace(" R50-FPN", ""), (xs[-1], ys[-1]), textcoords="offset points",
                          xytext=(-2, 5), ha="right", fontsize=6.5, color=INK_2)
        ax_t.set_xscale("log")
        ax_t.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
        ax_t.set_xlabel("fleet electricity cost (kEUR per year)")
        ax_t.set_ylabel("accuracy (mAP or top-1, %)")
        ax_t.set_title("c  accuracy vs operating cost", loc="left")

    be = a[a.precision != "fp32"].dropna(subset=["breakeven_eur_per_error"])
    be = be[be.breakeven_eur_per_error > 0]
    shown = [p for p in precisions if p != "fp32"]
    det_models = [m for m in models if m in ("frcnn_r50_fpn", "yolov10s", "yolov10x")]   # road-user errors
    for i, p in enumerate(shown):
        g = be[be.precision == p].set_index("model").reindex(det_models)
        y = np.arange(len(det_models)) + (i - (len(shown) - 1) / 2) * 0.22
        ax_b.scatter(g.breakeven_eur_per_error * 1e6, y, color=PRECISION_COLORS[p], s=26, zorder=2)
    ax_b.set_xscale("log")
    ax_b.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax_b.set_yticks(np.arange(len(det_models)), [MODEL_LABELS[m].replace(" R50-FPN", "") for m in det_models])
    ax_b.set_ylim(len(det_models) - 0.5, -0.5)
    ax_b.set_xlabel("break-even penalty ($\\mu$EUR per vanished road-user detection)")
    ax_b.set_title("d  penalty at which the saving vanishes", loc="left")
    fig.tight_layout(h_pad=1.6)
    save(fig, out / "R6_aibo")


def fig_noise_model(cfg, out, n=400_000):
    """Methods figure: analytic quantization-noise model (lines) and Monte Carlo check (markers).

    a: INT8 SQNR versus the clipping range alpha/sigma for Gaussian and Laplacian signals (granular plus
       clipping noise); FP8 E4M3 for reference. b: SQNR of a signal attenuated by a factor c after the
       scales were calibrated on the unattenuated signal (static per-tensor scales), with the contrast
       factors of the five COCO-C contrast severities.
    """
    import torch
    from scipy.stats import norm
    rng = np.random.default_rng(cfg["seed"])
    signals = {"Gaussian": rng.standard_normal(n), "Laplacian": rng.laplace(0, 1 / np.sqrt(2), n)}
    fp8_db = -10 * np.log10(0.180 * 2.0 ** -8)
    fp16_db = -10 * np.log10(0.180 * 2.0 ** -22)

    def sqnr(x, q):
        return 10 * np.log10(np.sum(x ** 2) / np.sum((q - x) ** 2))

    def int8(x, alpha):
        step = alpha / 127
        return np.clip(np.round(x / step), -127, 127) * step

    def fp8(x, alpha):
        scale = alpha / 448
        return (torch.from_numpy(x / scale).to(torch.float8_e4m3fn).double().numpy()) * scale

    def model_int8(kappa, dist):
        granular = (kappa / 127) ** 2 / 12
        if dist == "Gaussian":
            tail = 2 * norm.sf(kappa)
            clip = 2 * ((1 + kappa ** 2) * norm.sf(kappa) - kappa * norm.pdf(kappa))
        else:                                   # Laplace with unit variance: b = 1/sqrt(2)
            b = 1 / np.sqrt(2)
            tail = np.exp(-kappa / b)
            clip = 2 * b ** 2 * np.exp(-kappa / b)
        return -10 * np.log10(granular * (1 - tail) + clip)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9.0, 3.4))
    kappas = np.geomspace(1.5, 128, 200)
    styles = {"Gaussian": ("-", "o"), "Laplacian": ("--", "s")}
    for dist, x in signals.items():
        ls, mk = styles[dist]
        ax_a.plot(kappas, model_int8(kappas, dist), ls, color=PRECISION_COLORS["int8"], linewidth=1.6,
                  label=f"INT8, {dist} signal")
        ks = np.array([2, 3, 4, 6, 8, 12, 16, 32, 64])
        ax_a.plot(ks, [sqnr(x, int8(x, k)) for k in ks], mk, color=PRECISION_COLORS["int8"], markersize=4,
                  markerfacecolor="white")
    ax_a.axhline(fp8_db, color=PRECISION_COLORS["fp8"], linewidth=1.6, label="FP8 E4M3 (any range)")
    ax_a.plot([4, 16, 64], [sqnr(signals["Gaussian"], fp8(signals["Gaussian"], k)) for k in (4, 16, 64)], "o",
              color=PRECISION_COLORS["fp8"], markersize=4, markerfacecolor="white")
    ax_a.set_xscale("log")
    ax_a.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax_a.set_xlabel(r"clipping range $\kappa=\alpha/\sigma$")
    ax_a.set_ylabel("SQNR (dB)")
    ax_a.set_ylim(0, 50)
    ax_a.set_title("a  range calibration", loc="left")
    ax_a.legend(fontsize=7, loc="lower left")

    c = np.geomspace(1e-3, 1, 200)
    kappa_cal = 8.0
    x = signals["Gaussian"]
    alpha = kappa_cal * x.std()
    ax_b.plot(c, model_int8(kappa_cal, "Gaussian") + 20 * np.log10(c), color=PRECISION_COLORS["int8"],
              linewidth=1.6, label=r"INT8, static scale ($\kappa_\mathrm{cal}=8$)")
    ax_b.axhline(fp8_db, color=PRECISION_COLORS["fp8"], linewidth=1.6, label="FP8 E4M3, static scale")
    ax_b.axhline(fp16_db, color=PRECISION_COLORS["fp16"], linewidth=1.6, label="FP16")
    cs = np.array([1, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0.003])
    ax_b.plot(cs, [sqnr(v * x, int8(v * x, alpha)) for v in cs], "o", color=PRECISION_COLORS["int8"],
              markersize=4, markerfacecolor="white")
    ax_b.plot(cs, [sqnr(v * x, fp8(v * x, alpha)) for v in cs], "o", color=PRECISION_COLORS["fp8"],
              markersize=4, markerfacecolor="white")
    for sev, cf in enumerate([0.4, 0.3, 0.2, 0.1, 0.05], start=1):
        ax_b.axvline(cf, color=GRID_LINE, linewidth=0.8, zorder=0)
        ax_b.annotate(str(sev), (cf, 1.5), ha="center", fontsize=7, color=INK_2)
    ax_b.annotate("contrast severity", (0.14, 5), ha="center", fontsize=7, color=INK_2)
    ax_b.set_xscale("log")
    ax_b.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax_b.set_xlabel(r"signal attenuation $c=\sigma/\sigma_\mathrm{cal}$")
    ax_b.set_ylim(0, 80)
    ax_b.set_title("b  signal level after calibration", loc="left")
    ax_b.legend(fontsize=7, loc="center left", bbox_to_anchor=(0.0, 0.62))
    fig.tight_layout()
    save(fig, out / "M_noise_model")


CONTRAST_FACTORS = {1: 0.4, 2: 0.3, 3: 0.2, 4: 0.1, 5: 0.05}     # imagecorruptions contrast()


def static_scale_table(cfg, name, quant, severities=(1, 3, 5)):
    """Per convolution: median SQNR on clean images and its change under contrast reduction (same images)."""
    d = results_dir(cfg, "activations")
    paths = {s: d / f"{name}_{quant}_contrast{s}.csv.gz" for s in severities}
    clean_p = d / f"{name}_{quant}.csv.gz"
    if not clean_p.exists() or not all(p.exists() for p in paths.values()):
        return None
    cols = ["image", "tensor", "op", "order", "sqnr_db"]
    cond = {s: pd.read_csv(p, usecols=cols).query("op == 'Conv'") for s, p in paths.items()}
    ids = set(cond[severities[0]].image.unique())
    clean = pd.read_csv(clean_p, usecols=cols).query("op == 'Conv'")
    clean = clean[clean.image.isin(ids)]
    base = clean.groupby("order").sqnr_db.median()
    orders = sorted(base.index)
    out = pd.DataFrame({"order": orders, "layer": range(len(orders)), "clean": base.reindex(orders).values})
    for s, df in cond.items():
        out[f"d{s}"] = df.groupby("order").sqnr_db.median().reindex(orders).values - out.clean.values
    return out


def fig_static_scale(cfg, out, models=("yolov10s", "yolov10x"), severities=(1, 3, 5)):
    """Layer-wise test of Eq. (static): SQNR change under contrast reduction for INT8 and FP8."""
    from matplotlib import colormaps
    tabs = {(m, q): static_scale_table(cfg, m, q, severities) for m in models for q in ("int8fp32", "fp8")}
    if any(t is None for t in tabs.values()):
        return
    shades = colormaps[SEQUENTIAL](np.linspace(0.45, 0.95, len(severities)))
    fig, axes = plt.subplots(len(models), 2, figsize=(9.0, 2.7 * len(models)), sharey=True, squeeze=False)
    rows = []
    for i, m in enumerate(models):
        for j, q in enumerate(("int8fp32", "fp8")):
            ax, t = axes[i][j], tabs[(m, q)]
            for k, s in enumerate(severities):
                c = CONTRAST_FACTORS[s]
                ax.plot(t.layer, t[f"d{s}"], color=shades[k], linewidth=1.2,
                        label=f"severity {s} (c = {c:g})" if (i, j) == (0, 0) else None)
                ax.axhline(20 * np.log10(c), color=shades[k], linewidth=0.8, linestyle="--")
                first = t[f"d{s}"].iloc[: max(1, len(t) // 10)].median()
                rows.append({"model": m, "precision": q, "severity": s, "predicted_db": 20 * np.log10(c),
                             "first_decile_db": first, "median_db": t[f"d{s}"].median(),
                             "last_decile_db": t[f"d{s}"].iloc[-max(1, len(t) // 10):].median()})
            ax.axhline(0, color=MUTED, linewidth=0.8)
            ax.set_title(f"{'abcd'[2 * i + j]}  {MODEL_LABELS[m]}, {'INT8' if q.startswith('int8') else 'FP8'}",
                         loc="left")
            if j == 0:
                ax.set_ylabel("$\\Delta$SQNR vs clean (dB)")
            if i == len(models) - 1:
                ax.set_xlabel("convolution index (forward order)")
            ax.grid(axis="x", visible=False)
    axes[0][0].legend(fontsize=7, loc="lower right")
    fig.text(0.5, -0.01, "dashed: prediction $20\\log_{10}c$ of Eq. (static) for a static INT8 scale",
             ha="center", fontsize=7, color=INK_2)
    fig.tight_layout()
    pd.DataFrame(rows).to_csv(results_dir(cfg, "tables") / "static_scale.csv", index=False)
    save(fig, out / "S_static_scale")


CONDITION_LABELS = {"clean": "clean", "fog": "fog", "snow": "snow", "frost": "frost", "dark": "low light",
                    "brightness": "brightness (glare)", "contrast": "low contrast",
                    "gaussian_noise": "sensor noise", "motion_blur": "motion blur"}


def fig_conditions(cfg, out, image_id=336232, severity=3):
    """The eight adverse conditions applied to one COCO val2017 traffic scene."""
    from PIL import Image
    root = cfg["coco"]["root"]
    paths = {"clean": root / "val2017" / f"{image_id:012d}.jpg"}
    for c in ("fog", "snow", "frost", "dark", "brightness", "contrast", "gaussian_noise", "motion_blur"):
        paths[c] = root / "coco_c" / c / str(severity) / f"{image_id}.png"
    if not all(p.exists() for p in paths.values()):
        return
    fig, axes = plt.subplots(3, 3, figsize=(7.0, 5.05))
    for ax, (c, p) in zip(axes.flat, paths.items()):
        ax.imshow(Image.open(p).convert("RGB"))
        ax.set_title(CONDITION_LABELS[c] + ("" if c == "clean" else f", severity {severity}"), fontsize=8)
        ax.set_axis_off()
    fig.subplots_adjust(wspace=0.04, hspace=0.18, left=0.01, right=0.99, top=0.95, bottom=0.01)
    save(fig, out / "M_conditions")


def fig_energy_surface(cfg, out, name="efficientnet_b0", strategy="pepai_iter"):
    """Relative depth x measured energy per image x median MRE_proj over the selective variants.

    Variants of the iterative strategy (plus the fully quantized base), ordered by their measured energy;
    depth is binned (40 bins) and MRE_proj shown on a logarithmic axis so that the few very fragile
    layers do not hide the rest of the surface."""
    sel_p = results_dir(cfg, "tables") / "selective.csv"
    if not sel_p.exists():
        return
    sel = pd.read_csv(sel_p)
    sel = sel[(sel.model == name) & sel.energy_mj_per_img_bs8.notna()
              & ((sel.strategy == strategy) | ((sel.strategy == "pepai") & (sel.k == 0)))]
    d = results_dir(cfg, "selective")
    tag = {"pepai": "pepai_k", "pepai_iter": "iter_k"}
    frames = []
    for _, r in sel.iterrows():
        f = d / f"{name}_{tag[r.strategy]}{int(r.k)}_layers.csv"
        if f.exists():
            L = pd.read_csv(f)
            L = L[L.op == "Conv"].sort_values("order")
            L["depth"] = np.minimum((np.arange(len(L)) / len(L) * 40).astype(int), 39) / 39
            frames.append(L.assign(energy=r.energy_mj_per_img_bs8, k=int(r.k)))
    if len(frames) < 3:
        return
    df = pd.concat(frames)
    piv = df.pivot_table(index=["energy", "k"], columns="depth", values="mre_proj", aggfunc="median").sort_index()
    piv = piv.T.rolling(3, center=True, min_periods=1).median().T
    z = np.log10(np.clip(piv.values * 100, 0.1, None))
    X, Y = np.meshgrid(piv.columns.values, piv.index.get_level_values("energy").values)
    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.plot_surface(X, Y, z, cmap=SEQUENTIAL, linewidth=0.15, edgecolor=BLUE_700, alpha=0.95, rstride=1, cstride=1)
    ticks = [0.3, 1, 3, 10, 30]
    ax.set_zticks(np.log10(ticks), [f"{t:g}" for t in ticks])
    ax.set_xlabel("relative depth")
    ax.set_ylabel("GPU energy per image (mJ)")
    ax.set_zlabel("median MRE$_{proj}$ (%)")
    ax.set_title(f"{MODEL_LABELS[name]}: iterative selective-precision variants", fontsize=9)
    ax.view_init(elev=22, azim=-35)
    save(fig, out / "R3_energy_surface")

if __name__ == "__main__":
    cfg = load_config()
    style()
    out = results_dir(cfg, "figures")
    for f in (fig_activation_maps, fig_hexbin, fig_propagation, fig_operator_amplification, fig_speed,
              fig_selective, fig_surface, fig_robustness, fig_risk, fig_aibo, fig_energy_surface,
              fig_layer_profile, fig_conditions, fig_noise_model, fig_static_scale):
        f(cfg, out)
        print("done:", f.__name__, flush=True)
