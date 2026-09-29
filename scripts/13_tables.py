"""LaTeX table fragments (booktabs) for the manuscript, written to results/tables/tex/."""
import json
import re

import pandas as pd

from pepai.activations import load_activation_table
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


def write(path, body, cols, caption, label, notes=None):
    lines = [r"\begin{table}[h]", rf"\caption{{{caption}}}\label{{{label}}}",
             rf"\begin{{tabular}}{{@{{}}{cols}@{{}}}}", r"\toprule", *body, r"\botrule", r"\end{tabular}"]
    if notes:
        lines.append(rf"\footnotetext{{{notes}}}")
    lines.append(r"\end{table}")
    path.write_text("\n".join(lines) + "\n")


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
    body = [r"Model & Task & Input & Params (M) & GFLOPs & Conv layers & \multicolumn{3}{c}{Weights (MB)} \\",
            r"\cmidrule{7-9}", r" & & & & & & FP32 & FP16 & INT8 \\", r"\midrule"]
    for m in [m for m in ORDER if m in df.index]:
        r = df.loc[m]
        body.append(f"{MODEL_LABELS[m]} & {'detection' if r.task == 'det' else 'classification'} & {r.input} & "
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
    purpose = {"COCO train2017 calibration subset": "INT8 calibration (detectors)",
               "COCO val2017 (accuracy, deviation-risk)": "mAP, deviation--risk",
               "COCO val2017 analysis subset (activations, robustness)": "activations, robustness",
               "ImageNetV2 calibration": "INT8 calibration (classifiers)",
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
    bench = pd.read_csv(bench_p).groupby(["model", "precision", "batch"]).median(numeric_only=True)
    body = [r"Model & Precision & Accuracy (\%) & $\Delta$ vs FP32 [95\% CI] & p50 bs1 (ms) & p99 bs1 (ms) & "
            r"Throughput bs8 (img/s) & Energy bs8 (mJ/img) \\", r"\midrule"]
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
                delta = signed(a - ref)
                if ci is not None and (m, p) in ci.index:
                    r = ci.loc[(m, p)]
                    delta += f" [{signed(100 * r.delta_lo)}, {signed(100 * r.delta_hi)}]"
            name = MODEL_LABELS[m] if first else ""
            first = False
            body.append(
                f"{name} & {PRECISION_LABELS[p]} & {a:.1f} & {delta} & "
                f"{fmt(b1.graph_p50_ms if b1 is not None else None, 2)} & "
                f"{fmt(b1.graph_p99_ms if b1 is not None else None, 2)} & "
                f"{fmt(b8.graph_throughput_img_s if b8 is not None else None, 0)} & "
                f"{fmt(b8.energy_j_per_img * 1000 if b8 is not None else None, 2)} \\\\")
        body.append(r"\midrule")
    body = body[:-1]
    write(out / "R1_accuracy_speed.tex", body, "llrlrrrr",
          "Task accuracy, latency, throughput and energy per precision (TensorRT 11.3, NVIDIA H200)", "tab:accuracy",
          "Accuracy: COCO val2017 box mAP@[.5:.95] (5\\,000 images) for detectors, ImageNetV2 top-1 (9\\,000 images) "
          "for classifiers; 95\\% CI of the difference from 1\\,000 paired bootstrap resamples of the images. "
          "Latency: median (p50) and 99th percentile (p99) of 2\\,000 executions with CUDA graphs; throughput "
          "from the mean latency with CUDA graphs. Energy: gross GPU energy per image "
          "from the NVML counter at sustained load. Median of three repeats. INT8 and FP8 engines keep non-quantized "
          "operations in FP16; Faster R-CNN: backbone and FPN only.")


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
          f"which approximates batched processing of the synchronised camera frames of a vehicle; electricity price {a['electricity_eur_per_kwh'][1]} EUR/kWh and grid "
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


if __name__ == "__main__":
    cfg = load_config()
    out = results_dir(cfg, "tables", "tex")
    for f in (table_models, table_datasets, table_energy_reference, table_placement, table_accuracy_speed, table_layer_stats, table_aibo,
              table_hardware, table_engines, table_localization, table_propagation, table_selective):
        f(cfg, out)
        print("done:", f.__name__, flush=True)
