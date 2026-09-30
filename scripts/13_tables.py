"""LaTeX table fragments (booktabs) for the manuscript, written to results/tables/tex/."""
import json
import re

import pandas as pd

from pepai.activations import load_activation_table
from pepai.bench import benchmark_medians
from pepai.config import load_config, results_dir
from pepai.plots import MODEL_LABELS, PRECISION_LABELS

ORDER = ["frcnn_r50_fpn", "yolov10s", "yolov10x", "efficientnet_b0", "densenet121"]
SPECS_TASK = {"frcnn_r50_fpn": "det", "yolov10s": "det", "yolov10x": "det", "efficientnet_b0": "cls",
              "densenet121": "cls"}

# Horowitz, ISSCC 2014 (45 nm, 0.9 V), energy per operation in pJ.
HOROWITZ = [("8-bit integer add", 0.03), ("32-bit integer add", 0.1), ("16-bit float add", 0.4),
            ("32-bit float add", 0.9), ("8-bit integer multiply", 0.2), ("32-bit integer multiply", 3.1),
            ("16-bit float multiply", 1.1), ("32-bit float multiply", 3.7),
            ("32-bit SRAM read (8 kB)", 5.0), ("32-bit DRAM read", 640.0)]


# Tables wider than the text block go on a landscape page (Springer template: sidewaystable) in a smaller font.
WIDE = {"tab:accuracy", "tab:layerstats", "tab:aibo", "tab:engines", "tab:localization", "tab:propagation",
        "tab:selective", "tab:placement", "tab:operating", "tab:duplicates", "tab:calibvar", "tab:calibmethods", "tab:deployment"}
SMALL = {"tab:models", "tab:datasets"}


def write(path, body, cols, caption, label, notes=None):
    env = "sidewaystable" if label in WIDE else "table"
    size = (r"\footnotesize\setlength{\tabcolsep}{4pt}" if label in WIDE
            else (r"\footnotesize\setlength{\tabcolsep}{4pt}" if label in SMALL else ""))
    lines = [rf"\begin{{{env}}}" + ("" if env == "sidewaystable" else "[h]"), size,
             rf"\caption{{{caption}}}\label{{{label}}}",
             rf"\begin{{tabular}}{{@{{}}{cols}@{{}}}}", r"\toprule", *body, r"\botrule", r"\end{tabular}"]
    if notes:
        lines.append(rf"\footnotetext{{{notes}}}")
    lines.append(rf"\end{{{env}}}")
    path.write_text("\n".join(l for l in lines if l) + "\n")


def fmt(v, digits=1):
    return "--" if pd.isna(v) else f"{v:.{digits}f}"


def sig(v):
    """Two significant digits for small values, no spurious decimals for large ones."""
    if pd.isna(v):
        return "--"
    return f"{v:.0f}" if abs(v) >= 100 else (f"{v:.1f}" if abs(v) >= 1 else f"{v:.2g}")


def signed(v, digits=1):
    """Signed number with a typographic minus; values that round to zero are printed unsigned."""
    if pd.isna(v):
        return "--"
    r = round(v, digits)
    if r == 0:
        return f"{0:.{digits}f}"
    return ("$+$" if r > 0 else "$-$") + f"{abs(r):.{digits}f}"


def table_models(cfg, out):
    p = results_dir(cfg, "tables") / "models.csv"
    if not p.exists():
        return
    df = pd.read_csv(p).set_index("model")
    body = [r"Model & Task & Input & Params (M) & GFLOPs & Convs & \multicolumn{3}{c}{Weights (MB)} \\",
            r"\cmidrule{7-9}", r" & & & & & & FP32 & FP16 & INT8 \\", r"\midrule"]
    for m in [m for m in ORDER if m in df.index]:
        r = df.loc[m]
        body.append(f"{MODEL_LABELS[m]} & {'det.' if r.task == 'det' else 'cls.'} & {r.input.replace('x', '$\\times$')} & "
                    f"{fmt(r.params_M)} & {fmt(r.gflops)} & {int(r.conv_layers)} & {fmt(r.weights_mb_fp32)} & "
                    f"{fmt(r.weights_mb_fp16)} & {fmt(r.weights_mb_int8)} \\\\")
    write(out / "M1_models.tex", body, "lllrrrrrr", "Evaluated architectures", "tab:models",
          "Faster R-CNN: backbone and FPN only (the part that is quantized); GFLOPs at the benchmark input size. "
          "INT8 weight size counts quantized weights at 1 byte and the remaining tensors in FP16.")


def table_datasets(cfg, out):
    p = results_dir(cfg, "tables") / "datasets.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    body = [r"Split & Purpose & Images & Objects & Obj./img & S/M/L (\%) & Traffic (\%) \\",
            r"\midrule"]
    purpose = {"COCO train2017 calibration subset": "calibration (detectors)",
               "COCO val2017 (accuracy, deviation-risk)": "mAP, deviation--risk",
               "COCO val2017 analysis subset (activations, robustness)": "activations, robustness",
               "ImageNetV2 calibration": "calibration (classifiers)",
               "ImageNetV2 evaluation": "top-1, activations"}
    for _, r in df.iterrows():
        name = (r.split.split(" (")[0].replace("COCO ", "").replace(" calibration subset", " subset")
                .replace(" analysis subset", " first 500").replace("ImageNetV2", "INV2"))
        if pd.isna(r.objects):
            body.append(f"{name} & {purpose[r.split]} & {int(r.images):,} & -- & -- & -- & -- \\\\")
        else:
            body.append(f"{name} & {purpose[r.split]} & {int(r.images):,} & {int(r.objects):,} & {r.objects_per_image:.1f} & "
                        f"{r.small_pct:.0f} / {r.medium_pct:.0f} / {r.large_pct:.0f} & {r.traffic_objects_pct:.0f} \\\\")
    write(out / "M2_datasets.tex", [re.sub(r"(\d),(\d)", r"\1\\,\2", b) for b in body], "llrrrrr",
          "Data splits", "tab:datasets",
          "Traffic: share of objects in the classes person, bicycle, car, motorcycle, bus, truck, traffic light and "
          "stop sign. S/M/L: COCO size classes by box area ($<32^2$, $32^2$--$96^2$, $>96^2$ pixels). All subsets are drawn with a fixed seed.")


def table_energy_reference(cfg, out):
    body = [r"Operation & Energy (pJ) \\", r"\midrule"]
    body += [f"{op} & {e:g} \\\\" for op, e in HOROWITZ]
    write(out / "M5_energy_reference.tex", body, "lr",
          r"Energy per operation at 45\,nm (Horowitz, ISSCC 2014)", "tab:energyref")


def table_accuracy_speed(cfg, out):
    acc_p, bench_p = results_dir(cfg, "tables") / "accuracy.csv", results_dir(cfg, "tables") / "benchmark.csv"
    if not (acc_p.exists() and bench_p.exists()):
        return
    acc = pd.read_csv(acc_p).set_index(["model", "precision"])
    ci_p = results_dir(cfg, "tables") / "accuracy_ci.csv"
    ci = pd.read_csv(ci_p).set_index(["model", "precision"]) if ci_p.exists() else None
    bench = benchmark_medians(results_dir(cfg, "tables"))
    body = [r"Model & Precision & Accuracy (\%) & $\Delta$ vs FP32 [95\% CI] & p50 bs1 (ms) & p99 bs1 (ms) & "
            r"Images/s bs8 & mJ/image bs8 \\", r"\midrule"]
    for m in [m for m in ORDER if (m, "fp32") in acc.index]:
        metric = "mAP" if "mAP" in acc.columns and pd.notna(acc.loc[(m, "fp32")].get("mAP")) else "top1"
        ref = acc.loc[(m, "fp32"), metric] * 100
        first = True
        for p in ["fp32", "fp16", "int8", "fp8"]:
            if (m, p) not in acc.index:
                continue
            a = acc.loc[(m, p), metric] * 100
            b1 = bench.loc[(m, p, 1)] if (m, p, 1) in bench.index else None
            b8 = bench.loc[(m, p, 8)] if (m, p, 8) in bench.index else None
            delta = "--"
            if p != "fp32":
                delta = signed(a - ref, 2)
                if ci is not None and (m, p) in ci.index:
                    r = ci.loc[(m, p)]
                    delta += f" [{signed(100 * r.delta_lo, 2)}, {signed(100 * r.delta_hi, 2)}]"
            name = MODEL_LABELS[m] if first else ""
            first = False
            body.append(
                f"{name} & {PRECISION_LABELS[p]} & {a:.1f} & {delta} & "
                f"{fmt(b1.graph_p50_ms if b1 is not None else None, 2)} & "
                f"{fmt(b1.graph_p99_ms if b1 is not None else None, 2)} & "
                f"{fmt(b8.graph_throughput_img_s if b8 is not None else None, 0)} & "
                f"{sig(b8.energy_j_per_img * 1000) if b8 is not None else '--'} \\\\")
        body.append(r"\midrule")
    body = body[:-1]
    write(out / "R1_accuracy_speed.tex", body, "llrlrrrr",
          "Task accuracy, latency, throughput and energy per precision (TensorRT 11.3, NVIDIA H200)", "tab:accuracy",
          "Accuracy: COCO val2017 box mAP@[.5:.95] (5\\,000 images) for detectors, ImageNetV2 top-1 (9\\,000 images) "
          "for classifiers; 95\\% CI of the difference from 1\\,000 paired bootstrap resamples of the images. "
          "Latency: median (p50) and 99th percentile (p99) of 2\\,000 executions with CUDA graphs; throughput "
          "from the mean latency with CUDA graphs. Energy: gross GPU energy per image "
          "from the NVML counter during sustained CUDA-graph execution. Median of three repeats. INT8 and FP8 engines keep non-quantized "
          "operations in FP16; INT8 uses the entropy calibration of the toolchain on the full calibration set in its "
          "stored order (sensitivity: Table~\\ref{tab:calibvar}); Faster R-CNN: backbone and FPN only.")


def table_layer_stats(cfg, out, quant="int8fp32"):
    """Head-input and all-layer deviation per model (INT8 with FP32 fallback vs strict FP32)."""
    d = results_dir(cfg, "activations")
    body = [r"Model & \multicolumn{3}{c}{Head input (INT8)} & \multicolumn{2}{c}{All convolutions (INT8)} & "
            r"Head SQNR FP16 \\",
            r"\cmidrule{2-4}\cmidrule{5-6}",
            r" & MRE$_\text{proj}$ (\%) & MRE$_\text{elem}$ (\%) & SQNR (dB) & MRE$_\text{elem}$ (\%) & "
            r"worst SQNR (dB) & (dB) \\", r"\midrule"]
    any_rows = False
    for m in ORDER:
        p = d / f"{m}_{quant}.csv.gz"
        if not p.exists():
            continue
        df = load_activation_table(p)
        fin = df[df.final].groupby("image")[["mre_proj", "mre_elem", "sqnr_db"]].mean()
        conv = df[df.op == "Conv"]
        all_layers = conv.groupby("image").mre_elem.median()        # per image: median over all convolutions
        worst = conv.groupby("tensor").sqnr_db.median().min()
        f16 = d / f"{m}_fp16.csv.gz"
        f16_sqnr = (load_activation_table(f16).query("final").groupby("image").sqnr_db.mean().median()
                    if f16.exists() else float("nan"))

        def iqr(x, scale=1, digits=1):
            return (f"{x.median() * scale:.{digits}f} "
                    f"[{x.quantile(0.25) * scale:.{digits}f}, {x.quantile(0.75) * scale:.{digits}f}]")
        body.append(f"{MODEL_LABELS[m]} & {iqr(fin.mre_proj, 100)} & {iqr(fin.mre_elem, 100)} & {iqr(fin.sqnr_db)} & "
                    f"{iqr(all_layers, 100)} & {worst:.1f} & {fmt(f16_sqnr)} \\\\")
        any_rows = True
    if any_rows:
        write(out / "R2_layer_stats.tex", body, "lrrrrrr",
              "Activation-level deviation of INT8 from strict FP32 (500 images per model)", "tab:layerstats",
              "Head input: tensors consumed by the task head. Median over images, interquartile range in brackets. "
              "All convolutions: per image, the median element-wise relative error over every convolution output, "
              "comparable to the cumulative error of the conference paper (9--20\\%). Worst SQNR: lowest per-layer "
              "median. INT8 with FP32 fallback, so the deviation is caused by quantization alone; the FP16 column "
              "gives the numerical floor of the comparison.")


def table_aibo(cfg, out):
    p = results_dir(cfg, "tables") / "aibo.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    df = df[df.energy_basis == "energy_j_per_img"]
    body = [r"Model & Precision & Energy & Cost & CO$_2$ & \multicolumn{2}{c}{Saved vs FP32} & Critical errors & "
            r"Break-even \\",
            r"\cmidrule{6-7}",
            r" & & (MWh/yr) & (kEUR/yr) & (t/yr) & (kEUR/yr) & (t CO$_2$/yr) & (per $10^3$ images) & "
            r"(\textmu EUR/error) \\", r"\midrule"]
    for m in [m for m in ORDER if m in set(df.model)]:
        first = True
        for p_ in ["fp32", "fp16", "int8", "fp8"]:
            r = df[(df.model == m) & (df.precision == p_)]
            if r.empty:
                continue
            r = r.iloc[0]
            saved = "--" if p_ == "fp32" else f"{r.eur_saved_ref / 1000:.0f}"
            saved_t = "--" if p_ == "fp32" else f"{r.tco2_saved_ref:.0f}"
            crit = "--" if p_ == "fp32" else fmt(1000 * r.critical_per_image, 1)
            be = "--" if p_ == "fp32" or pd.isna(r.breakeven_eur_per_error) else sig(1e6 * r.breakeven_eur_per_error)
            body.append(f"{MODEL_LABELS[m] if first else ''} & {PRECISION_LABELS[p_]} & {r.kwh_per_year / 1000:.0f} & "
                        f"{r.eur_per_year_ref / 1000:.0f} & {r.tco2_per_year_ref:.0f} & {saved} & {saved_t} & "
                        f"{crit} & {be} \\\\")
            first = False
        body.append(r"\midrule")
    a = cfg["aibo"]
    write(out / "R3_aibo.tex", body[:-1], "llrrrrrrr", "AIBO fleet scenario: annual inference energy, cost, "
          "emissions and break-even error penalty per precision", "tab:aibo",
          f"{a['fleet_size']:,} vehicles, {a['cameras_per_vehicle']} cameras at {a['fps']} Hz, "
          f"{a['hours_per_day']} h/day; measured gross GPU energy per image at batch size {a['batch']} (NVIDIA H200), "
          f"i.e.\\ the synchronised frames of all cameras of a vehicle; electricity price {a['electricity_eur_per_kwh'][1]} EUR/kWh and grid "
          f"intensity {a['grid_gco2_per_kwh'][1]} g CO$_2$/kWh. Critical errors: FP32 road-user detections that "
          "vanish in the deployed engine (score-tolerant definition); classifiers: top-1 disagreements with FP32. "
          "Break-even: penalty per critical error at which the precision costs as much as FP32."
          .replace(",", "\\,", 1))


def table_placement(cfg, out):
    abl_p, sel_p = results_dir(cfg, "tables") / "placement_ablation.csv", results_dir(cfg, "tables") / "label_free_selection.csv"
    if not (abl_p.exists() and sel_p.exists()):
        return
    abl = pd.read_csv(abl_p)
    sel = pd.read_csv(sel_p).set_index(["model", "variant"])
    names = {"default": "default", "bn_fp16": "BN in FP16", "calib_max": "max calibration",
             "bn_fp16_calib_max": "BN in FP16 + max calibration"}
    body = [r"Model & Variant & Accuracy (\%) & Head SQNR (dB) & FP32 consistency & Picked by \\", r"\midrule"]
    for m in [m for m in ORDER if m in set(abl.model)]:
        g = abl[abl.model == m]
        acc_col = "top1" if SPECS_TASK[m] == "cls" else "mAP"
        s_m = sel.loc[m] if m in sel.index.get_level_values(0) else None
        pick_sqnr = s_m.head_sqnr_db.idxmax() if s_m is not None else None
        pick_cons = s_m.consistency.idxmax() if s_m is not None else None
        best = g.loc[g[acc_col].idxmax()].variant
        for i, (_, r) in enumerate(g.iterrows()):
            picks = [lab for lab, v in (("SQNR", pick_sqnr), ("consistency", pick_cons), ("accuracy", best))
                     if v == r.variant]
            cons = s_m.loc[r.variant].consistency if s_m is not None and r.variant in s_m.index else float("nan")
            body.append(f"{MODEL_LABELS[m] if i == 0 else ''} & {names.get(r.variant, r.variant)} & "
                        f"{100 * r[acc_col]:.1f} & {r.head_sqnr_db:.1f} & {fmt(cons, 3)} & {', '.join(picks)} \\\\")
        body.append(r"\midrule")
    write(out / "R_placement.tex", body[:-1], "llrrrl",
          "INT8 quantization variants and the variant selected by each label-free criterion", "tab:placement",
          "Accuracy: top-1 (classifiers) or box mAP (detectors) on the evaluation data. Head SQNR and FP32 "
          "consistency are computed on the calibration split only. \\emph{accuracy} marks the variant a test-set "
          "oracle would pick.")


def table_hardware(cfg, out):
    info_p = results_dir(cfg, "tables") / "gpu_info.json"
    if info_p.exists():
        info = json.loads(info_p.read_text())
    else:
        from pepai.bench import GPUMonitor
        info = GPUMonitor().info()
    import onnxruntime
    import tensorrt
    import torch
    import modelopt
    rows = [("GPU", f"{info.get('name', 'NVIDIA H200')}, power limit {info.get('power_limit_w', 700):.0f} W"),
            ("GPU driver", info.get("driver", "")),
            ("CPU", "Intel Xeon Platinum 8480C"),
            ("TensorRT", tensorrt.__version__), ("NVIDIA ModelOpt", modelopt.__version__),
            ("ONNX Runtime", onnxruntime.__version__), ("PyTorch / CUDA", f"{torch.__version__} / {torch.version.cuda}")]
    body = [r"Component & Version / specification \\", r"\midrule"] + [f"{k} & {v} \\\\" for k, v in rows]
    write(out / "M3_hardware.tex", body, "ll", "Hardware and software environment", "tab:hardware")


def table_engines(cfg, out):
    """Extended data: what the engines execute (engine inspector) and the resulting memory traffic."""
    p = results_dir(cfg, "tables") / "memory_traffic.csv"
    if not p.exists():
        return
    df = pd.read_csv(p).set_index(["model", "precision"])
    body = [r"Model & Precision & Kernels & Reformats & \multicolumn{3}{c}{Convolutions executed in} & "
            r"Activations & Weights & Traffic \\",
            r"\cmidrule{5-7}",
            r" & & & & INT8 & FP8 & FP16 & (MB) & (MB) & vs FP32 \\", r"\midrule"]
    for m in [m for m in ORDER if m in df.index.get_level_values(0)]:
        first = True
        for p_ in ["fp32", "fp16", "int8", "fp8"]:
            if (m, p_) not in df.index:
                continue
            r = df.loc[(m, p_)]
            body.append(f"{MODEL_LABELS[m] if first else ''} & {PRECISION_LABELS[p_]} & {r.kernels:.0f} & "
                        f"{r.reformats:.0f} & {r.conv_on_int8:.0f} & {r.conv_on_fp8:.0f} & {r.conv_on_fp16:.0f} & "
                        f"{r.activation_mb:.0f} & {r.weight_mb:.1f} & {fmt(r.traffic_vs_fp32, 2)} \\\\")
            first = False
        body.append(r"\midrule")
    write(out / "X_engines.tex", body[:-1], "llrrrrrrrr",
          "Composition and memory traffic per inference of the batch-1 TensorRT engines (engine inspector)", "tab:engines",
          "Kernels: launched layers after fusion. Reformats: layout or data-type conversions, including the copies "
          "that implement concatenations. Convolutions executed in: arithmetic precision of each convolution from "
          "its kernel name and input data type (FP32 engines: none of the three). Traffic: bytes of every layer "
          "input and output in its stored data type plus the deployed weights, i.e.\\ DRAM traffic without "
          "cross-layer caching.")


def head_sqnr(cfg, name, quant="int8fp32"):
    """Median over images of the mean head-input SQNR (as in table_layer_stats)."""
    path = results_dir(cfg, "activations") / f"{name}_{quant}.csv.gz"
    if not path.exists():
        return float("nan")
    df = load_activation_table(path)
    return df[df.final].groupby("image").sqnr_db.mean().median()


def table_propagation(cfg, out):
    """Extended data: injected noise per quantizer (FP32 statistics only) versus the measured head fidelity."""
    q_p, s_p = results_dir(cfg, "tables") / "quantizer_snr.csv", results_dir(cfg, "tables") / "quantizer_snr_summary.csv"
    if not (q_p.exists() and s_p.exists()):
        return
    q, summ = pd.read_csv(q_p), pd.read_csv(s_p).set_index("model")
    body = [r"Model & Quantizers & Injected SQNR (dB) & Model error (dB) & SQNR$_\text{add}$ (dB) & "
            r"SQNR$_h$ (dB) & $\bar\Gamma$ \\", r"\midrule"]
    rows = []
    for m in [m for m in ORDER if m in summ.index]:
        g = q[(q.model == m) & q.upstream_of_head.astype(bool)]
        measured = head_sqnr(cfg, m)
        add = summ.loc[m, "additive_gamma1_sqnr_db"]
        gamma = 10 ** ((add - measured) / 10)
        rows.append({"model": m, "gamma_bar": gamma, "sqnr_add_db": add, "sqnr_head_db": measured})
        inj = g.sqnr_inj_db
        body.append(f"{MODEL_LABELS[m]} & {len(g)} & {inj.median():.1f} [{inj.quantile(0.25):.1f}, "
                    f"{inj.quantile(0.75):.1f}] & {summ.loc[m, 'pred_vs_inj_mae_db']:.1f} & {add:.1f} & "
                    f"{measured:.1f} & {gamma:.2f} \\\\".replace("& -", "& $-$"))
    pd.DataFrame(rows).to_csv(results_dir(cfg, "tables") / "propagation_factor.csv", index=False)
    write(out / "X_propagation.tex", body, "lrrrrrr",
          "Noise injected by the individual INT8 activation quantizers and its accumulation at the head input",
          "tab:propagation",
          "Quantizers upstream of the head-input tensors. Injected SQNR: exact SQNR of each quantizer applied with "
          "its calibrated scale to the FP32 activations of 32 analysis images (median and interquartile range). "
          "Model error: mean absolute difference between Eq.~\\eqref{eq:sqnr-int8} and the injected SQNR for "
          "quantizers without clipping. SQNR$_\\text{add}$ and $\\bar\\Gamma$: Eq.~\\eqref{eq:gammabar}; "
          "SQNR$_h$: measured head-input SQNR (500 images, Table~\\ref{tab:layerstats}).")


SELECTIVE_ROWS = {   # (strategy, k) shown per model; random subsets are summarised over seeds
    "efficientnet_b0": [("pepai", 0), ("pepai", 1), ("pepai", 8), ("pepai", 20), ("pepai", 30), ("pepai_iter", 4),
                        ("pepai_iter", 8), ("pepai_iter", 20), ("noise", 8), ("noise", 20), ("random", 20),
                        ("depthwise", None)],
    "yolov10s": [("pepai", 0), ("pepai", 1), ("pepai", 8), ("pepai", 12), ("pepai", 20), ("noise", 8), ("noise", 20)],
    "yolov10x": [("pepai", 0), ("pepai", 1), ("pepai", 8), ("pepai", 12), ("pepai", 20)],
}
STRATEGY_LABELS = {"pepai": "PEP-AI one-shot", "pepai_iter": "PEP-AI iterative", "noise": "FP32 noise ranking",
                   "random": "random (3 seeds)", "depthwise": "all depthwise"}


def table_selective(cfg, out):
    p = results_dir(cfg, "tables") / "selective.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    acc = pd.read_csv(results_dir(cfg, "tables") / "accuracy.csv").set_index(["model", "precision"])
    body = [r"Model & Strategy & $k$ ($k_\text{eff}$) & Accuracy (\%) & p50 bs8 (ms) & Energy bs8 (mJ/img) \\",
            r"\midrule"]
    for m, spec in SELECTIVE_ROWS.items():
        g = df[df.model == m]
        if g.empty:
            continue
        metric = "top1" if SPECS_TASK[m] == "cls" else "mAP"
        first = True
        for strat, k in spec:
            h = g[(g.strategy == strat) & ((g.k == k) if k is not None else True)]
            if h.empty:
                continue
            label = "INT8 (base)" if (strat, k) == ("pepai", 0) else STRATEGY_LABELS[strat]
            kk = int(h.k.iloc[0])
            keff = h.k_effective.iloc[0] if "k_effective" in h and pd.notna(h.k_effective.iloc[0]) else kk
            if strat == "random":
                a = f"{100 * h[metric].mean():.1f} [{100 * h[metric].min():.1f}, {100 * h[metric].max():.1f}]"
            else:
                a = f"{100 * h[metric].iloc[0]:.1f}"
            ktxt = f"{kk}" if int(keff) == kk else f"{kk} ({int(keff)})"
            body.append(f"{MODEL_LABELS[m] if first else ''} & {label} & {ktxt} & {a} & "
                        f"{h.p50_ms_bs8.mean():.2f} & {h.energy_mj_per_img_bs8.mean():.1f} \\\\")
            first = False
        for ref in ("fp16", "fp32"):
            if (m, ref) in acc.index:
                body.append(f" & {PRECISION_LABELS[ref]} (reference) & -- & {100 * acc.loc[(m, ref), metric]:.1f} & & \\\\")
        body.append(r"\midrule")
    write(out / "X_selective.tex", body[:-1], "llrrrr",
          "Selective precision: accuracy, latency and energy versus the number $k$ of convolutions kept in FP16",
          "tab:selective",
          "Accuracy: ImageNetV2 top-1 (EfficientNet-B0) or COCO val2017 mAP (YOLOv10). $k_\\text{eff}$: FP16 "
          "convolutions including those added by the kernel-closure rule. Latency and energy from a shorter protocol "
          "than Table~\\ref{tab:accuracy} (500 executions, 5\\,s energy window); compare within this table. "
          "Random subsets: mean and range over three seeds.")


def table_localization(cfg, out):
    p = results_dir(cfg, "tables") / "localization.csv"
    if not p.exists() or p.stat().st_size < 10:
        return
    df = pd.read_csv(p)
    body = [r"Model & Precision & Class & Matched & \multicolumn{2}{c}{Bottom-edge shift (\% of box height)} & "
            r"\multicolumn{2}{c}{Distance error (\%)} & $>5$\% \\",
            r"\cmidrule{5-6}\cmidrule{7-8}",
            r" & & & detections & median & p95 & p95 & p99 & (\% of boxes) \\", r"\midrule"]
    for m in [m for m in ORDER if m in set(df.model)]:
        first = True
        for p_ in ["fp16", "int8", "fp8"]:
            for c in ["person", "car"]:
                r = df[(df.model == m) & (df.precision == p_) & (df["class"] == c)]
                if r.empty:
                    continue
                r = r.iloc[0]
                body.append(f"{MODEL_LABELS[m] if first else ''} & {PRECISION_LABELS[p_]} & {c} & "
                            f"{r.n_matched:,.0f} & {r.shift_rel_median_pct:.2f} & {r.shift_rel_p95_pct:.2f} & "
                            f"{r.dz_rel_p95_pct:.2f} & {r.dz_rel_p99_pct:.2f} & {100 * r.share_dz_gt_5pct:.1f} \\\\"
                            .replace(",", "\\,"))
                first = False
        body.append(r"\midrule")
    write(out / "X_localization.tex", body[:-1], "lllrrrrrr",
          "Monocular distance error caused by the FP32-to-deployed box shift (COCO val2017, 5\\,000 images)",
          "tab:localization",
          "Same-class matches (IoU $\\geq 0.5$) of FP32 detections with a score of at least 0.5. Distance error from "
          "Eq.~\\eqref{eq:distance} for a camera height of 1.65\\,m and object heights of 1.7\\,m (person) and "
          "1.5\\,m (car).")


def table_calibration_variability(cfg, out):
    """INT8 accuracy for seeded calibration subsets (halves; quarters for the classifiers) against the full set."""
    p = results_dir(cfg, "tables") / "calibration_variability.csv"
    acc_p = results_dir(cfg, "tables") / "accuracy.csv"
    if not (p.exists() and acc_p.exists()):
        return
    v = pd.read_csv(p)
    acc = pd.read_csv(acc_p).set_index(["model", "precision"])
    gam_p = results_dir(cfg, "tables") / "propagation_factor.csv"
    gam = pd.read_csv(gam_p).set_index("model") if gam_p.exists() else None
    body = [r"Model & $\bar\Gamma$ & Calibration images & Full set & Seed 1 & Seed 2 & Seed 3 & Range & "
            r"Max $|\Delta|$ \\", r"\midrule"]
    for m in [m for m in ORDER if m in set(v.model)]:
        metric = "top1" if SPECS_TASK[m] == "cls" else "mAP"
        full = acc.loc[(m, "int8"), metric] * 100 if (m, "int8") in acc.index else float("nan")
        g_all = v[v.model == m]
        g_bar = ""
        if gam is not None and m in gam.index:
            col = "gamma_bar" if "gamma_bar" in gam.columns else gam.columns[-1]
            g_bar = f"{float(gam.loc[m, col]):.2f}"
        first = True
        for n, g in sorted(g_all.groupby("n_calib"), key=lambda t: -t[0]):
            if len(g) < 2:
                continue
            vals = g.sort_values("seed")[metric].values * 100
            seeds = [f"{x:.2f}" for x in vals] + ["--"] * (3 - len(vals))
            body.append(f"{MODEL_LABELS[m] if first else ''} & {g_bar if first else ''} & {int(n)} & "
                        f"{full:.2f} & " + " & ".join(seeds) + f" & {vals.min():.2f}--{vals.max():.2f} & "
                        f"{abs(vals - full).max():.2f} \\\\")
            first = False
    write(out / "X_calibration.tex", body, "lrrrrrrrr",
          "Sensitivity of the deployed INT8 accuracy to the calibration sample", "tab:calibvar",
          "Entropy calibration on seeded random subsets of the calibration set (COCO train2017: 512 images; "
          "ImageNetV2 calibration split: 1\\,000 images), same exclusions and deployment engines as the main "
          "results; accuracy on the full evaluation sets (COCO val2017 box mAP, ImageNetV2 top-1, \\%). Full set: "
          "main result (Table~\\ref{tab:accuracy}); recalibrating on the full set with the same code reproduces it. "
          "Max $|\\Delta|$: largest deviation of a subset calibration from the full-set result.")


def table_operating_point(cfg, out):
    """Road-user detection quality per precision: AP50 per class and recall at the operating threshold."""
    p = results_dir(cfg, "tables") / "operating_point.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    body = [r"Model & Precision & \multicolumn{2}{c}{AP$_{50}$ (\%)} & \multicolumn{2}{c}{AP (\%)} & "
            r"\multicolumn{3}{c}{Recall at score $\geq0.5$ (\%)} & Precision (\%) \\",
            r"\cmidrule{3-4}\cmidrule{5-6}\cmidrule{7-9}",
            r" & & person & car & person & car & all & $\geq32^2$\,px & large & \\", r"\midrule"]
    for m in [m for m in ORDER if m in set(d.model)]:
        first = True
        for prec in ["fp32", "fp16", "int8", "fp8"]:
            g = d[(d.model == m) & (d.precision == prec)]
            if g.empty:
                continue
            r = g.iloc[0]
            body.append(f"{MODEL_LABELS[m] if first else ''} & {PRECISION_LABELS[prec]} & "
                        f"{100 * r.AP50_person:.1f} & {100 * r.AP50_car:.1f} & {100 * r.AP_person:.1f} & "
                        f"{100 * r.AP_car:.1f} & {100 * r.recall_road_all:.1f} & {100 * r.recall_road_ml:.1f} & "
                        f"{100 * r.recall_road_large:.1f} & {100 * r.precision_road_all:.1f} \\\\")
            first = False
        body.append(r"\midrule")
    write(out / "X_operating_point.tex", body[:-1], "llrrrrrrrr",
          "Detection quality for road users (COCO val2017, 5\\,000 images)", "tab:operating",
          "AP$_{50}$: average precision at IoU 0.5; AP: COCO AP@[.5:.95] of the class. Recall and precision: all "
          "road-user classes (person, bicycle, car, motorcycle, bus, truck) at the operating threshold (score "
          "$\\geq0.5$) and IoU $\\geq0.5$, with the greedy score-ordered matching of COCOeval; $\\geq32^2$\\,px: COCO medium "
          "and large objects. Precision is a lower bound, because COCO does not annotate every visible object.")


def table_duplicates(cfg, out):
    """Duplicate detections of the NMS-free YOLOv10 head and the effect of appending class-wise NMS."""
    p = results_dir(cfg, "tables") / "duplicates.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    body = [r"Model & Precision & Detections $\geq0.5$ & Duplicates (\%) & mAP & mAP + NMS & "
            r"mAP$_\text{large}$ & mAP$_\text{large}$ + NMS \\", r"\midrule"]
    labels = {**PRECISION_LABELS, "int8fp32": "INT8 (FP32 fallback)"}
    for m in [m for m in ORDER if m in set(d.model)]:
        first = True
        for prec in ["fp32", "fp16", "int8", "int8fp32", "fp8"]:
            g = d[(d.model == m) & (d.precision == prec)]
            if g.empty:
                continue
            r = g.iloc[0]
            body.append(f"{MODEL_LABELS[m] if first else ''} & {labels[prec]} & {int(r.dets_above_threshold)} & "
                        f"{100 * r.duplicate_share:.2f} & {100 * r.mAP:.1f} & {100 * r.mAP_nms:.1f} & "
                        f"{100 * r.mAP_large:.1f} & {100 * r.mAP_large_nms:.1f} \\\\")
            first = False
        if m == "yolov10x" and (results_dir(cfg, "tables") / "selective_duplicates.csv").exists():
            sd = pd.read_csv(results_dir(cfg, "tables") / "selective_duplicates.csv").sort_values("k")
            for _, r in sd[sd.model == m].iterrows():
                body.append(f" & INT8, {int(r.k)} ranked conv.\\ in FP16 & {int(r.dets_above_threshold)} & "
                            f"{100 * r.duplicate_share:.2f} & {100 * r.mAP:.1f} & {100 * r.mAP_nms:.1f} & "
                            f"{100 * r.mAP_large:.1f} & {100 * r.mAP_large_nms:.1f} \\\\")
        body.append(r"\midrule")
    write(out / "X_duplicates.tex", body[:-1], "llrrrrrr",
          "Duplicate detections of the NMS-free YOLOv10 head (COCO val2017, 5\\,000 images)", "tab:duplicates",
          "Duplicates: detections with a score of at least 0.5 that overlap a higher-scoring detection of the same "
          "class with IoU $\\geq0.7$. + NMS: class-wise non-maximum suppression (IoU 0.7) appended to the engine "
          "output as post-processing; on the GPU it takes 0.16\\,ms per image at batch size 1 and 0.04\\,ms per image "
          "at batch size 8 (torchvision, 105--114 boxes per image). Ranked conv.: selective precision with the PEP-AI "
          "ranking (Table~\\ref{tab:selective}).")


def table_calibration_methods(cfg, out):
    """INT8 accuracy under the calibration variants: toolchain (stored order), first image varied, order-independent."""
    t = results_dir(cfg, "tables")
    if not all((t / f).exists() for f in ("accuracy.csv", "calibration_variability.csv", "global_entropy.csv")):
        return
    acc = pd.read_csv(t / "accuracy.csv").set_index(["model", "precision"])
    var = pd.read_csv(t / "calibration_variability.csv")
    glob = pd.read_csv(t / "global_entropy.csv").set_index("model")
    conv = pd.read_csv(t / "conv_only_placement.csv").set_index("variant") if (t / "conv_only_placement.csv").exists() \
        else None
    body = [r"Model & FP32 & \multicolumn{3}{c}{INT8, toolchain entropy calibration} & INT8, order-independent & FP8 \\",
            r"\cmidrule{3-5}",
            r" & & full set & subsets (range) & max $|\Delta|$ & entropy calibration & \\", r"\midrule"]
    for m in [m for m in ORDER if m in glob.index]:
        metric = "top1" if SPECS_TASK[m] == "cls" else "mAP"
        f = lambda p: acc.loc[(m, p), metric] * 100 if (m, p) in acc.index else float("nan")
        n_full = 1000 if SPECS_TASK[m] == "cls" else 512
        v = var[(var.model == m) & (var.n_calib < n_full)][metric] * 100
        full = f("int8")
        body.append(f"{MODEL_LABELS[m]} & {f('fp32'):.2f} & {full:.2f} & {v.min():.2f}--{v.max():.2f} & "
                    f"{(v - full).abs().max():.2f} & {glob.loc[m, metric] * 100:.2f} & {f('fp8'):.2f} \\\\")
    if conv is not None:
        body.append(f"EfficientNet-B0, convolution inputs only & {acc.loc[('efficientnet_b0', 'fp32'), 'top1'] * 100:.2f} & "
                    f"{conv.loc['toolchain', 'top1'] * 100:.2f} & -- & -- & {conv.loc['global', 'top1'] * 100:.2f} & -- \\\\")
    write(out / "X_calibration_methods.tex", body, "lrrrrrr",
          "INT8 accuracy under different range-setting procedures", "tab:calibmethods",
          "Toolchain: incremental entropy calibration of ModelOpt on ONNX Runtime (one image per step), full "
          "calibration set in stored order (main results) and seeded random subsets, which differ in their first "
          "image (Table~\\ref{tab:calibvar}). Order-independent: global 2048-bin histogram with NVIDIA's reference "
          "KL search (Supplementary Section~\\ref{sec:supp-quant}), applied to the same quantizers. Convolution inputs "
          "only: quantizers on the inputs of convolutions and matrix multiplications, as in~\\cite{wu2020integer}. "
          "COCO val2017 box mAP and ImageNetV2 top-1 (\\%); FP8: max calibration.")


def table_deployment(cfg, out):
    """Recommended deployment per network: accuracy, batch-8 latency and energy of the measured configurations."""
    t = results_dir(cfg, "tables")
    if not (t / "accuracy.csv").exists():
        return
    acc = pd.read_csv(t / "accuracy.csv").set_index(["model", "precision"])
    bench = benchmark_medians(t)
    dup = pd.read_csv(t / "duplicates.csv").set_index(["model", "precision"]) if (t / "duplicates.csv").exists() else None
    nms = pd.read_csv(t / "nms_cost.csv").set_index(["model", "batch"]) if (t / "nms_cost.csv").exists() else None
    pla = pd.read_csv(t / "placement_ablation.csv").set_index(["model", "variant"]) \
        if (t / "placement_ablation.csv").exists() else None
    # (model, configuration label, precision of the engine, accuracy override, extra batch-8 latency in ms)
    configs = {
        "frcnn_r50_fpn": [("FP16", "fp16", None, 0), ("INT8", "int8", None, 0)],
        "yolov10s": [("FP16", "fp16", None, 0), ("INT8", "int8", None, 0), ("FP8", "fp8", None, 0)],
        "yolov10x": [("FP16", "fp16", None, 0), ("FP8", "fp8", None, 0), ("INT8 + NMS", "int8", "nms", "nms")],
        "efficientnet_b0": [("FP16", "fp16", None, 0), ("FP8", "fp8", None, 0)],
        "densenet121": [("FP16", "fp16", None, 0), ("FP8", "fp8", None, 0),
                        ("INT8, BN in FP16", "int8bnfp16", "bn_fp16", 0)],
    }
    body = [r"Model & Configuration & Accuracy (\%) & $\Delta$ vs FP32 & p50 bs8 (ms) & Energy bs8 (mJ/img) & "
            r"Energy vs FP32 & Energy vs FP16 \\", r"\midrule"]
    for m, rows in configs.items():
        metric = "top1" if SPECS_TASK[m] == "cls" else "mAP"
        ref = acc.loc[(m, "fp32"), metric] * 100
        e32 = bench.loc[(m, "fp32", 8), "energy_j_per_img"]
        e16 = bench.loc[(m, "fp16", 8), "energy_j_per_img"]
        first = True
        for label, prec, override, extra in rows:
            if (m, prec, 8) not in bench.index:
                continue
            if override == "nms" and dup is not None:
                a = dup.loc[(m, "int8"), "mAP_nms"] * 100
            elif override == "bn_fp16" and pla is not None:
                a = pla.loc[(m, "bn_fp16"), metric] * 100
            else:
                a = acc.loc[(m, prec), metric] * 100
            lat = bench.loc[(m, prec, 8), "graph_p50_ms"] + (nms.loc[(m, 8), "p50_ms"] if extra == "nms" and nms is not None
                                                            else 0)
            e = bench.loc[(m, prec, 8), "energy_j_per_img"]
            body.append(f"{MODEL_LABELS[m] if first else ''} & {label} & {a:.1f} & {signed(a - ref, 1)} & {lat:.2f} & "
                        f"{1000 * e:.1f} & {signed(100 * (e / e32 - 1), 0)}\\% & "
                        f"{signed(100 * (e / e16 - 1), 0)}\\% \\\\")
            first = False
        body.append(r"\midrule")
    write(out / "X_deployment.tex", body[:-1], "llrrrrrr",
          "Deployment options per network on the H200 (batch size 8)", "tab:deployment",
          "Accuracy: COCO val2017 box mAP or ImageNetV2 top-1 of the deployed engine. INT8 + NMS: class-wise NMS "
          "appended to the engine output (its GPU time is included in the latency; its energy is not). INT8, BN in "
          "FP16: the placement selected by the label-free criteria (Table~\\ref{tab:placement}). Energy: gross GPU energy "
          "per image with CUDA graphs.")


if __name__ == "__main__":
    cfg = load_config()
    out = results_dir(cfg, "tables", "tex")
    for f in (table_models, table_datasets, table_energy_reference, table_placement, table_accuracy_speed, table_layer_stats, table_aibo,
              table_hardware, table_engines, table_localization, table_propagation, table_selective,
              table_calibration_variability, table_operating_point, table_duplicates,
              table_calibration_methods, table_deployment):
        f(cfg, out)
        print("done:", f.__name__, flush=True)
