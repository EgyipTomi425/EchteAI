"""LaTeX table fragments (booktabs) for the manuscript, written to results/tables/tex/."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from pepai.activations import load_activation_table
from pepai.bench import benchmark_medians
from pepai.config import CODE_ROOT, load_config, results_dir
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
        "tab:placement", "tab:duplicates", "tab:calibmethods", "tab:deployment", "tab:modelcheck", "tab:formulas",
        "tab:prediction", "tab:static", "tab:staticval", "tab:staticpct", "tab:extcheck"}
SMALL = {"tab:models", "tab:datasets"}
COMPACT = {"tab:selective", "tab:operating", "tab:calibvar"}   # fit the text width with narrower column gaps

# Reading guide placed at the start of the notes of the result tables (describes the reference results).
TAKEAWAY = {
    "tab:accuracy": "Compare each row with the FP32 row of the same network. FP16 is lossless for all five networks; "
                    "INT8 always needs the least energy per image, but its accuracy cost ranges from a quarter of a point "
                    "(Faster R-CNN) to 41 points (EfficientNet-B0), whereas FP8 stays within half a point of FP32.",
    "tab:layerstats": r"Ranking the networks by head-input SQNR gives the same order as ranking them by INT8 accuracy "
                      r"loss in Table~\ref{tab:accuracy}: the lower the SQNR, the larger the loss. The FP16 column lies "
                      r"far above every INT8 value, so the comparison itself adds no relevant error.",
    "tab:engines": r"The convolution columns show that every quantized convolution of the INT8 engines runs on INT8 "
                   r"kernels, and the last column how much data each engine moves: INT8 engines 14--26\% and FP8 "
                   r"engines 26--39\% of the FP32 bytes. In DenseNet-121, most INT8 and FP16 kernels are reformats "
                   r"(the copies of its concatenations), which the FP8 engine avoids.",
    "tab:propagation": r"The injected noise per quantizer is similar in all five networks (27--33\,dB); the networks "
                       r"differ in how it accumulates. SQNR$_\text{add}$ is the value expected if the noise simply added "
                       r"up: $\bar\Gamma<1$ means that the network damps the noise (Faster R-CNN, YOLOv10-S), "
                       r"$\bar\Gamma>1$ that it amplifies it (EfficientNet-B0).",
    "tab:selective": r"Compare the strategies at the same $k$. For EfficientNet-B0, the iterative ranking reaches the "
                     r"FP32 accuracy with 20 FP16 convolutions, far more than the FP32 noise ranking or random "
                     r"subsets; YOLOv10-X recovers most of its loss only at $k=20$, and YOLOv10-S gains nothing because "
                     r"its noise is spread over the whole network.",
    "tab:calibvar": r"The sensitivity is smallest for the network that damps the noise most (Faster R-CNN, "
                    r"$\bar\Gamma=0.13$) and largest for the one that amplifies it (EfficientNet-B0, up to 17 points), so "
                    r"networks with a large $\bar\Gamma$ should be calibrated and checked on several samples.",
    "tab:calibmethods": r"INT8 accuracy depends strongly on how the ranges are set: the order-independent procedure is "
                        r"much worse for the YOLOv10 networks and EfficientNet-B0 but better for DenseNet-121. FP8, whose "
                        r"range is simply the maximum, stays within half a point of FP32 for all five networks.",
    "tab:duplicates": r"Only YOLOv10-X in INT8 produces many duplicates (about 15\% of its confident detections); a "
                      r"class-wise NMS removes them and recovers most of the mAP, and so do 20 ranked FP16 convolutions. "
                      r"The FP32-fallback rows show that the effect is caused by quantization, not by FP16 arithmetic.",
    "tab:deployment": r"Within each network, choose the cheapest row whose accuracy loss is acceptable. Faster R-CNN can "
                      r"run in INT8 as it is; the YOLOv10 detectors lose little in FP8, and YOLOv10-X needs a class-wise "
                      r"NMS in INT8; DenseNet-121 is cheapest without loss in FP8; EfficientNet-B0 is best served by "
                      r"FP16, because its FP8 engine uses more energy.",
    "tab:prediction": r"Compare the two LOO columns with the measured loss. The prediction from the measured head-input "
                      r"SQNR (LOO$_h$) stays within about a factor of two of the measurement for all five networks, "
                      r"whereas the one from FP32 statistics alone (LOO$_\text{add}$) misses YOLOv10-X and "
                      r"EfficientNet-B0 by a factor of four to six, in opposite directions, because it cannot see "
                      r"$\bar\Gamma$. The FP8 loss is at most 1\% for every network.",
    "tab:modelcheck": r"Small differences mean that the model describes the measurement: the noise of single quantizers "
                      r"and the plateau of the layer SQNR agree within 1.4\,dB. The large difference in the head-input "
                      r"row is $10\log_{10}\bar\Gamma$, the propagation that FP32 statistics cannot provide, and the last "
                      r"row shows that contrast loss accumulates along the network beyond its injected effect.",
    "tab:static": r"The INT8 SQNR$_\text{add}$ estimated from the parameters orders the five measured networks with one "
                  r"exchange, but it lies 8--12.5\,dB above the measured value (last column), so it screens networks "
                  r"without predicting their deployment numbers. Per-channel weight scales gain 6--11\,dB over a single "
                  r"scale per tensor.",
    "tab:staticval": r"Start with the FP8 rows: static and measured quantizer SQNR agree within 0.2\,dB, so the FP8 "
                     r"noise follows from the parameters alone. For INT8 the static estimate is up to 6.4\,dB optimistic. "
                     r"The last two columns compare the measured relative loss with the one predicted from SQNR$_h$.",
    "tab:staticpct": r"Each FP8 quantizer adds about 2.6--2.7\% relative error in every network and each INT8 quantizer "
                     r"1--5\%, but the error at the head input ranges from about 5\% to more than 100\%: propagation, "
                     r"not the injected noise, decides the accuracy loss.",
    "tab:extcheck": r"In the rows within range, compare the measured loss with the prediction of the relation fitted on "
                    r"the five TensorRT networks; most predictions fall within the 95\% CI. $\bar\Gamma$ stays below 1 "
                    r"for the residual, dense and inception networks but spreads from 0.5 to almost 7 for the depthwise "
                    r"ones, so the block type alone does not determine it.",
    "tab:aibo": r"The savings scale with the FP32 energy of a network, so the large detectors dominate. A small "
                r"break-even value means that even a very low penalty per critical error outweighs the energy saving "
                r"(EfficientNet-B0 and DenseNet-121 in INT8).",
    "tab:placement": r"Compare the label-free picks with the accuracy pick: for Faster R-CNN and DenseNet-121 the criteria "
                     r"select the best or a nearly equal variant without labels, whereas for the YOLOv10 detectors they "
                     r"miss the better calibration method, which therefore needs a small labelled set.",
    "tab:localization": r"Look at the last column: FP16 moves almost no box beyond a 5\% distance error and the YOLOv10 "
                        r"engines at most 1\% of the boxes, whereas the INT8 and FP8 engines of Faster R-CNN move "
                        r"5--6\% of the person and car boxes beyond this limit.",
    "tab:operating": r"Apart from YOLOv10-X in INT8, AP$_{50}$ changes by less than one point. YOLOv10-X in INT8 keeps "
                     r"its recall but loses precision (93.6 against 80.2\%): the extra boxes are the duplicates of "
                     r"Table~\ref{tab:duplicates}.",
}


def write(path, body, cols, caption, label, notes=None):
    env = "sidewaystable" if label in WIDE else "table"
    size = (r"\footnotesize\setlength{\tabcolsep}{4pt}" if label in WIDE | SMALL
            else (r"\footnotesize\setlength{\tabcolsep}{2.5pt}" if label in COMPACT else ""))
    lines = [rf"\begin{{{env}}}" + ("" if env == "sidewaystable" else "[htbp]"), size,
             rf"\caption{{{caption}}}\label{{{label}}}",

             rf"\begin{{tabular}}{{@{{}}{cols}@{{}}}}", r"\toprule", *body, r"\botrule", r"\end{tabular}",
]
    if label in TAKEAWAY:
        notes = TAKEAWAY[label] + (" " + notes if notes else "")
    if notes:
        lines.append(rf"\footnotetext{{{notes}}}")
    lines.append(rf"\end{{{env}}}")
    path.write_text("\n".join(l for l in lines if l) + "\n")


def tex_sci(v):
    """p-value in print form: 0.04, or 3\times10^{-7} for small values."""
    if v >= 0.001:
        return f"{v:.2g}"
    mant, exp = f"{v:.0e}".split("e")
    return rf"{mant}\times10^{{{int(exp)}}}"


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
              "All convolutions: per image, the median element-wise relative error over every convolution output. "
              "Worst SQNR: lowest per-layer "
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
    try:
        import onnxruntime
        import tensorrt
        import torch
        import modelopt
        env = {"tensorrt": tensorrt.__version__, "modelopt": modelopt.__version__,
               "onnxruntime": onnxruntime.__version__, "torch": torch.__version__, "cuda": torch.version.cuda}
    except ImportError:  # offline regeneration without the GPU stack: versions recorded with the results
        env_p = Path(cfg["paths"]["results"]) / "environment.json"
        if not env_p.exists():
            env_p = CODE_ROOT / "reference_results" / "environment.json"
        env = json.loads(env_p.read_text())
    rows = [("GPU", f"{info.get('name', 'NVIDIA H200')}, power limit {info.get('power_limit_w', 700):.0f} W"),
            ("GPU driver", info.get("driver", "")),
            ("CPU", "Intel Xeon Platinum 8480C"),
            ("TensorRT", env["tensorrt"]), ("NVIDIA ModelOpt", env["modelopt"]),
            ("ONNX Runtime", env["onnxruntime"]), ("PyTorch / CUDA", f"{env['torch']} / {env['cuda']}")]
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
    if not path.exists():  # activations are not distributed; use the value stored with the results
        pf = results_dir(cfg, "tables") / "propagation_factor.csv"
        if quant == "int8fp32" and pf.exists():
            v = pd.read_csv(pf).set_index("model")["sqnr_head_db"]
            return float(v.get(name, float("nan")))
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


def table_formulas(cfg, out):
    """Extended data: the noise and propagation formulas per number format, with their mathematical status."""
    fit_p = results_dir(cfg, "tables") / "prediction_fit.csv"
    law = ""
    if fit_p.exists():
        f = pd.read_csv(fit_p).set_index("predictor")
        if "sqnr_head_db" in f.index and "exponent_r" in f.columns:
            r = f.loc["sqnr_head_db"]
            law = (rf"Relative loss vs head-input SQNR & \multicolumn{{3}}{{p{{9.1cm}}}}{{$\times10$ per {r.db_per_decade:.1f}\,dB, "
                   rf"i.e.\ $\propto r_h^{{{r.exponent_r:.2f}}}$ (95\% CI {r.exponent_r_lo:.2f}--{r.exponent_r_hi:.2f})}} & "
                   r"Section~\ref{sec:worked} & empirical, five networks \\")
    body = [r"Quantity & INT8 & FP8 (E4M3) & FP16 & Source & Status \\", r"\midrule",
            r"Scale and grid & $s=\alpha/127$, uniform, 255 levels & $s=\alpha/448$, $p=4$ significand bits & "
            r"no scale, $p=11$ & Eq.~\eqref{eq:int8} & definition \\",
            r"Injected noise of one quantizer & $\mathrm{SQNR}\approx52.9\,\mathrm{dB}-20\log_{10}\kappa$ & "
            r"$6.02p+7.44=31.5$\,dB & $6.02p+7.44=73.7$\,dB & Lemma~\ref{lem:noise} & proved under (A1), (A2) \\",
            r"Dependence on the range $\kappa=\alpha/\sigma$ & $-20$\,dB per decade; clipping above $\alpha$ & "
            r"none in the normal range & none & Eqs.~\eqref{eq:sqnr-int8}, \eqref{eq:sqnr-fp} & proved \\",
            r"Signal scaled by $c$ after calibration & $\Delta\mathrm{SQNR}=20\log_{10}c$ & $0$ for "
            r"$c\sigma\gg2^{-6}s$ & $0$ & Corollary~\ref{cor:static} & proved, no clipping \\",
            r"Head-input error & \multicolumn{3}{p{9.1cm}}{$r_h^2=\sum_n\Gamma_{n\to h}\rho_n^2$; "
            r"$\rho_n$ from the row above} & Proposition~\ref{prop:selective} & proved, linearised \\",
            r"Prediction from FP32 statistics & \multicolumn{3}{p{9.1cm}}{$\mathrm{SQNR}_\text{add}=-10\log_{10}"
            r"\sum_n\rho_n^2$ (all $\Gamma=1$)} & Eq.~\eqref{eq:gammabar} & bound-free estimate \\",
            r"Number $N$ of quantizers & \multicolumn{3}{p{9.1cm}}{$\mathrm{SQNR}_\text{add}=-10\log_{10}N-10\log_{10}"
            r"\langle\rho_n^2\rangle$: $-3$\,dB per doubling} & Eq.~\eqref{eq:gammabar} & identity \\",
            r"Propagation factor (one measurement) & \multicolumn{3}{p{9.1cm}}{$\bar\Gamma=10^{(\mathrm{SQNR}_\text{add}-"
            r"\mathrm{SQNR}_h)/10}$; $<1$ attenuating, $>1$ amplifying} & Eq.~\eqref{eq:gammabar} & definition \\",
            r"Plateau of a contracting chain & \multicolumn{3}{p{9.1cm}}{$\mathrm{SQNR}_\infty=-20\log_{10}\rho+"
            r"10\log_{10}(1-g^2)$ for $g<1$} & Proposition~\ref{prop:fixedpoint} & proved \\",
            r"Gain of a non-folded BN & \multicolumn{3}{p{9.1cm}}{closed form from $\gamma,\beta,\mu,v$} & "
            r"Eq.~\eqref{eq:bn} & derived, equal noise per channel \\",
            r"Layers to keep in FP16 & \multicolumn{3}{p{9.1cm}}{the $k$ largest contributions $\Gamma_{n\to h}\rho_n^2$} & "
            r"Proposition~\ref{prop:selective} & proved, linearised \\"]
    if law:
        body.append(law)
    write(out / "X_formulas.tex", body, "p{3.3cm}p{3.6cm}p{3.1cm}p{2.0cm}p{2.7cm}p{2.6cm}", "Noise and propagation formulas per number format", "tab:formulas",
          "$\\alpha$: clipping range, $\\sigma$: root-mean-square value of the activation, $\\kappa=\\alpha/\\sigma$, "
          "$p$: significand bits including the implicit one, $\\rho_n$: relative noise injected at node $n$, "
          "$\\langle\\cdot\\rangle$: mean over the $N$ quantizers, $\\mathrm{SQNR}_h$: measured head-input SQNR, "
          "$g$: propagation gain of a chain, $r_h$: relative error of the head input. Proofs: Supplementary "
          "Sections~\\ref{sec:supp-noise} and~\\ref{sec:supp-prop}. The empirical relation is a least-squares fit "
          "of $\\log_{10}$ of the relative loss on the head-input SQNR with a $t$-based confidence interval.")


def table_model_check(cfg, out, models=("yolov10s", "yolov10x"), severity=5):
    """Extended data: every step of the noise and propagation model computed for two detectors and compared with
    the measurement (all inputs are tables written by 06, 26 and 12_figures)."""
    t = results_dir(cfg, "tables")
    need = ["quantizer_snr.csv", "quantizer_snr_fp8.csv", "quantizer_snr_summary.csv", "propagation_factor.csv",
            "recursion_fit.csv", "static_scale.csv", f"quantizer_snr_contrast{severity}.csv",
            f"quantizer_snr_fp8_contrast{severity}.csv"]
    if not all((t / n).exists() for n in need):
        return
    q, q8 = pd.read_csv(t / "quantizer_snr.csv"), pd.read_csv(t / "quantizer_snr_fp8.csv")
    summ = pd.read_csv(t / "quantizer_snr_summary.csv").set_index("model")
    pf = pd.read_csv(t / "propagation_factor.csv").set_index("model")
    rec = pd.read_csv(t / "recursion_fit.csv").set_index(["model", "precision"])
    st = pd.read_csv(t / "static_scale.csv")
    qc, q8c = pd.read_csv(t / f"quantizer_snr_contrast{severity}.csv"), pd.read_csv(t / f"quantizer_snr_fp8_contrast{severity}.csv")

    def contrast_change(base, cond, m, int8):
        j = base[base.model == m].set_index("tensor").join(cond[cond.model == m].set_index("tensor"), rsuffix="_c",
                                                           how="inner")
        j = j[j.upstream_of_head.astype(bool)]
        pred = 20 * np.log10(j.rms_c / j.rms) if int8 else 0 * j.rms
        return float(np.median(pred)), float(np.median(j.sqnr_inj_db_c - j.sqnr_inj_db))

    rows, data = [], {}
    for m in models:
        up, up8 = q[(q.model == m) & q.upstream_of_head.astype(bool)], q8[(q8.model == m) & q8.upstream_of_head.astype(bool)]
        meas_layer = st[(st.model == m) & (st.precision == "int8fp32") & (st.severity == severity)].measured_median_db.dropna()
        data[m] = [
            (up.sqnr_pred_db.median(), up.sqnr_inj_db.median()),
            (up8.sqnr_pred_db.median(), up8.sqnr_inj_db.median()),
            (pf.loc[m, "sqnr_add_db"], pf.loc[m, "sqnr_head_db"]),
            (rec.loc[(m, "int8fp32"), "plateau_db"], rec.loc[(m, "int8fp32"), "measured_db"]),
            contrast_change(q, qc, m, True),
            contrast_change(q8, q8c, m, False),
            (contrast_change(q, qc, m, True)[0], float(meas_layer.iloc[0]) if len(meas_layer) else float("nan")),
        ]
    labels = [("INT8 noise of one quantizer, median SQNR", r"Eq.~\eqref{eq:sqnr-int8}"),
              ("FP8 noise of one quantizer, median SQNR", r"Eq.~\eqref{eq:sqnr-fp}"),
              (r"Head-input SQNR from FP32 statistics ($\Gamma=1$)", r"Eq.~\eqref{eq:gammabar}"),
              ("Plateau of the layer SQNR (chain fit)", r"Eq.~\eqref{eq:fixedpoint}"),
              (f"INT8 injected SQNR change, contrast severity {severity}", r"Eq.~\eqref{eq:static}"),
              (f"FP8 injected SQNR change, contrast severity {severity}", r"Eq.~\eqref{eq:static}"),
              (f"INT8 layer SQNR change (cumulative), severity {severity}", r"Eq.~\eqref{eq:static}")]
    head = " & ".join(rf"\multicolumn{{3}}{{c}}{{{MODEL_LABELS[m]}}}" for m in models)
    body = [rf"Quantity (dB) & Source & {head} \\",
            "".join(rf"\cmidrule{{{3 + 3 * i}-{5 + 3 * i}}}" for i in range(len(models))),
            " & & " + " & ".join(["predicted & measured & difference"] * len(models)) + r" \\", r"\midrule"]
    for i, (lab, src) in enumerate(labels):
        cells = []
        for m in models:
            pr, me = data[m][i]
            show = signed if "change" in lab else fmt
            cells += [show(pr, 1), show(me, 1), signed(pr - me, 1)]
            rows.append({"model": m, "quantity": lab, "predicted_db": pr, "measured_db": me, "difference_db": pr - me})
        body.append(f"{lab} & {src} & " + " & ".join(cells) + r" \\")
    pd.DataFrame(rows).to_csv(t / "model_check.csv", index=False)
    write(out / "X_model_check.tex", body, "ll" + "rrr" * len(models),
          "Worked example: the noise and propagation model against the measurement", "tab:modelcheck",
          "Medians over the quantizers upstream of the head input (32 analysis images) or over the layers (chain fit, "
          "500 images; contrast, 100 images). INT8 with FP32 fallback. Head input: the difference equals "
          "$10\\log_{10}\\bar\\Gamma$. Contrast rows: prediction $20\\log_{10}c$ with the measured attenuation $c$ "
          "of each quantizer input (INT8) and $0$ (FP8); the last row compares the injected change with the measured "
          "cumulative change of the layer SQNR, which also contains the shift of the operating point.")


PREDICTION_FIT_SOURCES = {"sqnr_add_db": "FP32 statistics only", "sqnr_head_db": "one quantized measurement"}


def exact_spearman_p(x, y):
    """One-sided exact p-value of Spearman's rho (all permutations; small n)."""
    from itertools import permutations
    rx, ry = pd.Series(x).rank().values, pd.Series(y).rank().values
    obs = np.corrcoef(rx, ry)[0, 1]
    vals = [np.corrcoef(rx, np.array(p))[0, 1] for p in permutations(ry)]
    return obs, float(np.mean([v >= obs - 1e-12 for v in vals])) if obs > 0 else float(np.mean([v <= obs + 1e-12 for v in vals]))


def table_prediction(cfg, out):
    """Extended data: prediction of INT8 tolerance for all five networks, from FP32 statistics alone and after one
    measurement of the quantized head input, with leave-one-out predictions of the relative INT8 loss."""
    t = results_dir(cfg, "tables")
    need = ["quantizer_snr.csv", "quantizer_snr_fp8.csv", "quantizer_snr_summary.csv", "quantizer_snr_fp8_summary.csv",
            "propagation_factor.csv", "accuracy.csv"]
    if not all((t / n).exists() for n in need):
        return
    q, q8 = pd.read_csv(t / "quantizer_snr.csv"), pd.read_csv(t / "quantizer_snr_fp8.csv")
    summ, summ8 = (pd.read_csv(t / n).set_index("model") for n in ("quantizer_snr_summary.csv",
                                                                   "quantizer_snr_fp8_summary.csv"))
    pf = pd.read_csv(t / "propagation_factor.csv").set_index("model")
    acc = pd.read_csv(t / "accuracy.csv").set_index(["model", "precision"])
    rows = []
    for m in [m for m in ORDER if m in pf.index]:
        up = q[(q.model == m) & q.upstream_of_head.astype(bool)]
        up8 = q8[(q8.model == m) & q8.upstream_of_head.astype(bool)]
        metric = "top1" if pd.notna(acc.loc[(m, "fp32")].get("top1")) else "mAP"
        ref = acc.loc[(m, "fp32"), metric]
        rows.append({"model": m, "quantizers": len(up),
                     "int8_pred_median_db": up.sqnr_pred_db.median(), "int8_exact_median_db": up.sqnr_inj_db.median(),
                     "int8_mae_db": summ.loc[m, "pred_vs_inj_mae_db"],
                     "fp8_pred_median_db": up8.sqnr_pred_db.median(), "fp8_exact_median_db": up8.sqnr_inj_db.median(),
                     "sqnr_add_db": pf.loc[m, "sqnr_add_db"], "sqnr_add_fp8_db": summ8.loc[m, "additive_gamma1_sqnr_db"],
                     "sqnr_head_db": pf.loc[m, "sqnr_head_db"], "gamma_bar": pf.loc[m, "gamma_bar"],
                     "int8_rel_loss": (ref - acc.loc[(m, "int8"), metric]) / ref,
                     "fp8_rel_loss": (ref - acc.loc[(m, "fp8"), metric]) / ref})
    d = pd.DataFrame(rows)
    fits = []
    y = np.log10(d.int8_rel_loss.values)
    for x_col, source in PREDICTION_FIT_SOURCES.items():
        x = d[x_col].values
        b, a = np.polyfit(x, y, 1)
        r2 = 1 - np.sum((y - (a + b * x)) ** 2) / np.sum((y - y.mean()) ** 2)
        loo = np.array([np.polyval(np.polyfit(np.delete(x, k), np.delete(y, k), 1), x[k]) for k in range(len(x))])
        d[f"loo_rel_loss_from_{x_col}"] = 10 ** loo
        from scipy import stats
        lr = stats.linregress(x, y)
        tq = stats.t.ppf(0.975, len(x) - 2)
        lo, hi = lr.slope - tq * lr.stderr, lr.slope + tq * lr.stderr
        rho, p = exact_spearman_p(-x, d.int8_rel_loss.values)
        fits.append({"predictor": x_col, "source": source, "slope_log10_per_db": b, "intercept": a,
                     "db_per_decade": -1 / b, "r2": r2, "spearman_rho": rho, "exact_p_one_sided": p,
                     "exponent_r": -20 * b, "exponent_r_lo": -20 * hi, "exponent_r_hi": -20 * lo,
                     "slope_p": lr.pvalue,
                     "loo_max_factor": float(np.max(10 ** np.abs(loo - y))),
                     "loo_median_factor": float(np.median(10 ** np.abs(loo - y)))})
    rho_g, p_g = exact_spearman_p(d.gamma_bar.values, d.int8_rel_loss.values)
    fits.append({"predictor": "gamma_bar", "source": "one quantized measurement", "spearman_rho": rho_g,
                 "exact_p_one_sided": p_g})
    d.to_csv(t / "prediction.csv", index=False)
    pd.DataFrame(fits).to_csv(t / "prediction_fit.csv", index=False)
    body = [r"Model & $N$ & \multicolumn{3}{c}{INT8 quantizer (dB)} & \multicolumn{2}{c}{SQNR$_\text{add}$ (dB)} & "
            r"SQNR$_h$ & $\bar\Gamma$ & \multicolumn{3}{c}{INT8 loss (\%)} & FP8 \\",
            r"\cmidrule{3-5}\cmidrule{6-7}\cmidrule{10-12}",
            r" & & Eq.~\eqref{eq:sqnr-int8} & exact & MAE & INT8 & FP8 & (dB) & & meas. & LOO$_\text{add}$ & "
            r"LOO$_h$ & loss (\%) \\", r"\midrule"]
    for _, r in d.iterrows():
        body.append(f"{MODEL_LABELS[r.model]} & {r.quantizers} & {fmt(r.int8_pred_median_db)} & "
                    f"{fmt(r.int8_exact_median_db)} & {fmt(r.int8_mae_db)} & "
                    f"{fmt(r.sqnr_add_db)} & {fmt(r.sqnr_add_fp8_db)} & {signed(r.sqnr_head_db) if r.sqnr_head_db < 0 else fmt(r.sqnr_head_db)} & "
                    f"{r.gamma_bar:.2f} & {100 * r.int8_rel_loss:.1f} & {100 * r.loo_rel_loss_from_sqnr_add_db:.1f} & "
                    f"{100 * r.loo_rel_loss_from_sqnr_head_db:.1f} & {100 * r.fp8_rel_loss:.1f} \\\\")
    f_add, f_head = fits[0], fits[1]
    write(out / "X_prediction.tex", body, "lrrrrrrrrrrrr",
          "Prediction of INT8 tolerance for the five networks", "tab:prediction",
          "Quantizer SQNR: median over the quantizers upstream of the head input; Eq.~\\eqref{eq:sqnr-int8} from the "
          "normalised range, exact from the calibrated scale applied to FP32 activations (32 images); MAE: mean "
          "absolute difference for quantizers without clipping. FP8: every quantizer within 0.02\\,dB of the 31.5\\,dB "
          "of Eq.~\\eqref{eq:sqnr-fp}. $N$: quantizers upstream of the head. "
          "SQNR$_\\text{add}$: Eq.~\\eqref{eq:gammabar} with unit propagation factors (FP32 statistics only); "
          "SQNR$_h$: measured head-input SQNR of the INT8 engine (one measurement). Relative loss: accuracy loss of "
          "the deployed engine divided by the FP32 accuracy (mAP or top-1). LOO$_\\text{add}$, LOO$_h$: leave-one-out prediction of "
          f"$\\log_{{10}}$ of the relative loss from a straight line fitted to the other four networks "
          f"(all five: {f_add['db_per_decade']:.1f}\\,dB per decade, $R^2={f_add['r2']:.2f}$ for SQNR$_\\text{{add}}$; "
          f"{f_head['db_per_decade']:.1f}\\,dB per decade, $R^2={f_head['r2']:.2f}$ for SQNR$_h$).")


STATIC_LABELS = {"mobilenet_v2": r"MobileNetV2$^\dagger$", "resnet50": r"ResNet-50$^\dagger$",
                 "mobilenet_v3_large": r"MobileNetV3-L$^\dagger$", "shufflenet_v2_x1_0": r"ShuffleNetV2$^\dagger$",
                 "mnasnet1_0": r"MnasNet$^\dagger$", "resnet18": r"ResNet-18$^\dagger$",
                 "regnet_y_800mf": r"RegNetY-800MF$^\dagger$", "googlenet": r"GoogLeNet$^\dagger$",
                 "resnet34": r"ResNet-34$^\dagger$", "efficientnet_b1": r"EfficientNet-B1$^\dagger$",
                 "densenet169": r"DenseNet-169$^\dagger$", "resnext50_32x4d": r"ResNeXt-50$^\dagger$",
                 "resnet101": r"ResNet-101$^\dagger$"}
VALIDATION_ORDER = ["mobilenet_v2", "resnet50", "resnet18", "resnet34", "resnet101", "resnext50_32x4d",
                    "regnet_y_800mf", "googlenet", "mobilenet_v3_large", "mnasnet1_0", "shufflenet_v2_x1_0",
                    "efficientnet_b1", "densenet169"]
FAMILY = {"resnet18": "residual", "resnet34": "residual", "resnet50": "residual", "resnet101": "residual",
          "resnext50_32x4d": "residual", "regnet_y_800mf": "residual", "googlenet": "inception",
          "mobilenet_v2": "depthwise", "mobilenet_v3_large": "depthwise", "mnasnet1_0": "depthwise",
          "shufflenet_v2_x1_0": "depthwise", "efficientnet_b0": "depthwise", "efficientnet_b1": "depthwise",
          "densenet121": "dense", "densenet169": "dense"}


def table_static(cfg, out):
    """Extended data: static analysis of the FP32 parameters (43_static_analysis.py), no data and no execution."""
    t = results_dir(cfg, "tables")
    if not (t / "static_analysis.csv").exists():
        return
    st = pd.read_csv(t / "static_analysis.csv")
    order = [m for m in ORDER if m in set(st.model)] + [m for m in st.model if m not in ORDER]
    st = st.set_index("model").loc[order]
    pred = pd.read_csv(t / "prediction.csv").set_index("model") if (t / "prediction.csv").exists() else None
    body = [r"Model & Params & DW & $N$ & \multicolumn{2}{c}{Ch.\ spread} & \multicolumn{2}{c}{Weights INT8} & "
            r"Act.\ INT8 & \multicolumn{2}{c}{SQNR$_\text{add}$} & BN & Meas. \\",
            r"\cmidrule{5-6}\cmidrule{7-8}\cmidrule{10-11}",
            r" & (M) & & & med. & max & per ch. & per t. & (dB) & INT8 & FP8 & gain & (dB) \\", r"\midrule"]
    for m, r in st.iterrows():
        label = MODEL_LABELS.get(m, STATIC_LABELS.get(m, m))
        bn = f"{r.bn_gain_median:.2f} ({int(r.nonfoldable_bn)})" if r.nonfoldable_bn else "--"
        meas = fmt(pred.loc[m, "sqnr_add_db"]) if pred is not None and m in pred.index else "--"
        body.append(f"{label} & {r.params_m:.1f} & {int(r.depthwise_convs)} & "
                    f"{int(r.act_quantizers)} & {r.act_channel_spread_median:.1f} & {r.act_channel_spread_max:.0f} & "
                    f"{fmt(r.w_int8_per_channel_median_db)} & {fmt(r.w_int8_per_tensor_median_db)} & "
                    f"{fmt(r.act_int8_sqnr_median_db)} & {fmt(r.static_sqnr_add_int8_db)} & "
                    f"{fmt(r.static_sqnr_add_fp8_db)} & {bn} & {meas} \\\\")
    note = ""
    if pred is not None:
        both = [m for m in pred.index if m in st.index]
        rho, p = exact_spearman_p(-st.loc[both, "static_sqnr_add_int8_db"].values, pred.loc[both, "int8_rel_loss"].values)
        note = (f" Ranking of the five networks of the article by static INT8 SQNR$_\\text{{add}}$ against their "
                f"relative INT8 loss: Spearman $\\rho={rho:.1f}$ (exact one-sided $p={p:.2f}$).")
        pd.DataFrame([{"static_rank_rho": rho, "p": p}]).to_csv(t / "static_rank.csv", index=False)
    write(out / "X_static.tex", body, "lrrrrrrrrrrrr", "Static analysis of the FP32 parameters", "tab:static",
          "Computed from the weights and the batch-normalisation buffers only (no image, no forward pass; "
          "\\texttt{43\\_static\\_analysis.py}). DW: depthwise convolutions. $N$: activation tensors modelled from "
          "batch normalisation. Channel spread: 90th/10th percentile of the channel RMS under one per-tensor scale. "
          "Weights: median exact SQNR over the layers, BN folded, per channel and per tensor (FP8 weights: "
          "31.7--32.3\\,dB for all networks). Act.\\ INT8: median predicted injected SQNR "
          "(Gaussian channels, MSE-optimal range). SQNR$_\\text{add}$: static, in dB. BN gain: median gain of "
          "Eq.~\\eqref{eq:bn} over the non-foldable batch normalisations (their number in brackets). "
          "Meas.: SQNR$_\\text{add}$ from FP32 activations and the calibrated scales of the toolchain "
          "(Table~\\ref{tab:prediction}). $^\\dagger$ not used elsewhere in this article (Table~\\ref{tab:staticval})." + note)


def table_static_validation(cfg, out):
    """Extended data: static prediction against simulated quantization on ImageNetV2 (44_validate_static.py)."""
    t = results_dir(cfg, "tables")
    if not (t / "static_validation.csv").exists():
        return
    v = pd.read_csv(t / "static_validation.csv")
    order = [m for m in VALIDATION_ORDER + ORDER if m in set(v.model)]
    v = v.set_index("model").loc[order]
    body = [r"Model & Format & \multicolumn{3}{c}{Quantizer SQNR, static $-$ measured} & "
            r"\multicolumn{2}{c}{SQNR$_\text{add}$ (dB)} & SQNR$_h$ & $\bar\Gamma$ & Top-1 & "
            r"\multicolumn{2}{c}{Relative loss (\%)} \\",
            r"\cmidrule{3-5}\cmidrule{6-7}\cmidrule{11-12}",
            r" & & median (dB) & MAE (dB) & rank $\rho$ & static & measured & (dB) & & (\%) & measured [95\% CI] & "
            r"from SQNR$_h$ \\", r"\midrule"]
    for m, r in v.iterrows():
        label = MODEL_LABELS.get(m, STATIC_LABELS.get(m, m))
        for k, fmt_ in enumerate(("int8", "fp8")):
            name = label if k == 0 else ""
            rho = f"{r[f'{fmt_}_static_vs_measured_spearman']:.2f}" if fmt_ == "int8" else "--"
            body.append(
                f"{name} & {fmt_.upper()} & {signed(r[f'{fmt_}_static_vs_measured_median_diff_db'])} & "
                f"{fmt(r[f'{fmt_}_static_vs_measured_mae_db'], 2 if fmt_ == 'fp8' else 1)} & {rho} & "
                f"{fmt(r[f'static_sqnr_add_{fmt_}_db'])} & {fmt(r[f'{fmt_}_sqnr_add_measured_db'])} & "
                f"{fmt(r[f'{fmt_}_sqnr_head_db'])} & {r[f'{fmt_}_gamma_bar']:.2f} & {100 * r[f'{fmt_}_top1']:.1f} & "
                f"{100 * r[f'{fmt_}_rel_loss']:.1f} [{100 * r[f'{fmt_}_rel_loss_lo']:.1f}, "
                f"{100 * r[f'{fmt_}_rel_loss_hi']:.1f}] & {100 * r[f'{fmt_}_rel_loss_predicted']:.1f} \\\\".replace("[-", "[$-$"))
        body.append(r"\midrule")
    n_eval, n_cal = int(v.n_eval.iloc[0]), int(v.n_calib.iloc[0])
    write(out / "X_static_validation.tex", body[:-1], "llrrrrrrrrrr",
          "Static analysis against simulated quantization", "tab:staticval",
          f"ImageNetV2, {n_cal} calibration and {n_eval} evaluation images, CPU simulation "
          "(\\texttt{44\\_validate\\_static.py}): symmetric per-tensor quantizers on every batch-normalised tensor "
          "(INT8: MSE-optimal range; FP8 E4M3: maximum), weights per output channel. Top-1 of FP32: "
          + ", ".join(f"{MODEL_LABELS.get(m, STATIC_LABELS.get(m, m))} {100 * v.loc[m, 'fp32_top1']:.1f}\\%" for m in order)
          + ". Quantizer SQNR: static prediction from the BN parameters against the exact injected SQNR on 64 "
          "evaluation images. SQNR$_\\text{add}$ includes the weight quantizers. Relative loss: measured with a paired "
          "bootstrap interval, and predicted from the measured SQNR$_h$ with the calibration of "
          "Table~\\ref{tab:prediction}. The simulated placement and calibration differ from the TensorRT toolchain, "
          "so the INT8 numbers of EfficientNet-B0 and DenseNet-121 differ from Table~\\ref{tab:accuracy}. "
          "$^\\dagger$ not used elsewhere in this article.")


def table_static_percent(cfg, out):
    """Extended data: the static and measured deviations of Table tab:staticval as relative errors in percent."""
    t = results_dir(cfg, "tables")
    if not (t / "static_validation.csv").exists():
        return
    v = pd.read_csv(t / "static_validation.csv")
    order = [m for m in VALIDATION_ORDER + ORDER if m in set(v.model)]
    v = v.set_index("model").loc[order]
    r = lambda db: f"{100 * 10 ** (-db / 20):.1f}"
    body = [r"Model & Format & \multicolumn{2}{c}{Injected per quantizer (\%)} & "
            r"\multicolumn{2}{c}{Injected, whole network (\%)} & Head input (\%) & Relative loss (\%) \\",
            r"\cmidrule{3-4}\cmidrule{5-6}",
            r" & & static & measured & static & measured & measured & measured \\", r"\midrule"]
    for m in order:
        x = v.loc[m]
        name = MODEL_LABELS.get(m, STATIC_LABELS.get(m, m))
        for f in ("int8", "fp8"):
            static_q = x[f"{f}_injected_median_db"] + x[f"{f}_static_vs_measured_median_diff_db"]
            body.append(f"{name if f == 'int8' else ''} & {f.upper()} & {r(static_q)} & {r(x[f'{f}_injected_median_db'])} & "
                        f"{r(x[f'static_sqnr_add_{f}_db'])} & {r(x[f'{f}_sqnr_add_measured_db'])} & "
                        f"{r(x[f'{f}_sqnr_head_db'])} & {100 * x[f'{f}_rel_loss']:.1f} \\\\")
        body.append(r"\midrule")
    write(out / "X_static_percent.tex", body[:-1], "llrrrrrr",
          "Static and measured deviation as relative error", "tab:staticpct",
          "Relative error $r=\\lVert e\\rVert/\\lVert f\\rVert=10^{-\\mathrm{SQNR}/20}$ in percent of the signal, from the "
          "SQNR values of Table~\\ref{tab:staticval}. Injected per quantizer: median over the quantizers; whole network: "
          "$\\mathrm{SQNR}_\\text{add}$ including the weights, with unit propagation factors; head input: measured at the "
          "input of the last linear layer, i.e.\\ after propagation. Static values need no image; measured values need "
          "unlabelled images, and only the relative loss needs labels. $^\\dagger$ not used elsewhere in this article.")


def table_extended_check(cfg, out):
    """Extended data: out-of-sample check of the propagation factor and of the loss relation on all networks of the
    simulated check (44_validate_static.py). Networks whose head-input SQNR lies outside the range of the five
    calibration networks are marked and left out of the summary statistics (outside the validity range)."""
    t = results_dir(cfg, "tables")
    if not (t / "static_validation.csv").exists() or not (t / "prediction.csv").exists():
        return
    from scipy import stats
    v = pd.read_csv(t / "static_validation.csv")
    order = [m for m in VALIDATION_ORDER + ORDER if m in set(v.model)]
    v = v.set_index("model").loc[order]
    cal = pd.read_csv(t / "prediction.csv")
    lo_db, hi_db = cal.sqnr_head_db.min(), cal.sqnr_head_db.max()
    rows, summary = [], []
    for m in order:
        for f in ("int8", "fp8"):
            x = v.loc[m]
            h = x[f"{f}_sqnr_head_db"]
            rows.append({"model": m, "family": FAMILY.get(m, "other"), "format": f, "fp32_top1": x["fp32_top1"],
                         "sqnr_head_db": h, "gamma_bar": x[f"{f}_gamma_bar"], "rel_loss": x[f"{f}_rel_loss"],
                         "rel_loss_lo": x[f"{f}_rel_loss_lo"], "rel_loss_hi": x[f"{f}_rel_loss_hi"],
                         "rel_loss_predicted": x[f"{f}_rel_loss_predicted"],
                         "in_range": bool(lo_db <= h <= hi_db)})
    d = pd.DataFrame(rows)
    d["within_ci"] = (d.rel_loss_lo <= d.rel_loss_predicted) & (d.rel_loss_predicted <= d.rel_loss_hi)
    d["factor"] = np.maximum(d.rel_loss_predicted, 1e-4) / np.maximum(d.rel_loss, 1e-4)
    d["factor"] = np.where(d.factor < 1, 1 / d.factor, d.factor)
    d.to_csv(t / "extended_check.csv", index=False)
    for f in ("int8", "fp8", "both"):
        e = d[d.in_range] if f == "both" else d[(d.format == f) & d.in_range]
        rho, p = stats.spearmanr(e.sqnr_head_db, e.rel_loss)
        big = e[e.rel_loss > 0.01]
        summary.append({"format": f, "n": len(e), "n_out_of_range": int((~d.in_range & ((d.format == f) | (f == "both"))).sum()),
                        "spearman_rho": rho, "p": p, "within_ci": int(e.within_ci.sum()),
                        "median_factor_loss_gt_1pct": float(big.factor.median()) if len(big) else np.nan,
                        "max_factor_loss_gt_1pct": float(big.factor.max()) if len(big) else np.nan, "n_loss_gt_1pct": len(big)})
    fam = d[d.in_range].groupby(["format", "family"]).gamma_bar.agg(["median", "min", "max", "count"]).reset_index()
    pd.DataFrame(summary).to_csv(t / "extended_summary.csv", index=False)
    fam.to_csv(t / "extended_family.csv", index=False)
    body = [r"Model & Block type & FP32 top-1 & Format & SQNR$_h$ (dB) & $\bar\Gamma$ & \multicolumn{2}{c}{Relative loss (\%)} & In range \\",
            r"\cmidrule{7-8}", r" & & (\%) & & & & measured [95\% CI] & from SQNR$_h$ & \\", r"\midrule"]
    for m in order:
        for _, r in d[d.model == m].iterrows():
            name = MODEL_LABELS.get(m, STATIC_LABELS.get(m, m)) if r.format == "int8" else ""
            fam_ = r.family if r.format == "int8" else ""
            top = f"{100 * r.fp32_top1:.1f}" if r.format == "int8" else ""
            body.append(f"{name} & {fam_} & {top} & {r.format.upper()} & {fmt(r.sqnr_head_db)} & {r.gamma_bar:.2f} & "
                        f"{100 * r.rel_loss:.1f} [{100 * r.rel_loss_lo:.1f}, {100 * r.rel_loss_hi:.1f}] & "
                        f"{100 * r.rel_loss_predicted:.1f} & {'yes' if r.in_range else 'no'} \\\\".replace("[-", "[$-$").replace("& -", "& $-$"))
    sm = pd.DataFrame(summary).set_index("format")
    fam_txt = "; ".join(f"{fm.upper()} {g.family}: {g['median']:.2f} ({g['min']:.2f}--{g['max']:.2f}, $n={int(g['count'])}$)"
                        for fm in ("int8", "fp8") for _, g in fam[fam.format == fm].iterrows())
    notes = (f"ImageNetV2, simulated quantization on the CPU (\\texttt{{44\\_validate\\_static.py}}), 128 calibration images; "
             f"evaluation on 2\\,000 images for the networks of Table~\\ref{{tab:staticval}} and on 1\\,000 images for the "
             f"others. From SQNR$_h$: predicted from the measured SQNR$_h$ with the relation of Section~\\ref{{sec:worked}}, calibrated on "
             f"the five TensorRT networks. In range: SQNR$_h$ within the range of the calibration networks "
             f"({lo_db:.1f} to {hi_db:.1f}\\,dB); rows outside it are reported but not used in the summary. Summary over the "
             f"rows in range: Spearman $\\rho$ between SQNR$_h$ and the measured loss ${sm.loc['both','spearman_rho']:.2f}$ "
             f"(negative: lower SQNR, larger loss; $n={int(sm.loc['both','n'])}$, $p={tex_sci(sm.loc['both','p'])}$); prediction within the 95\\% CI in "
             f"{int(sm.loc['both','within_ci'])} of {int(sm.loc['both','n'])} cases; for losses above 1\\%, median miss "
             f"$\\times${sm.loc['both','median_factor_loss_gt_1pct']:.2f}, largest $\\times${sm.loc['both','max_factor_loss_gt_1pct']:.2f}. "
             f"$\\bar\\Gamma$ by block type, median (range): {fam_txt}. $^\\dagger$ not used elsewhere in this article.")
    write(out / "X_extended_check.tex", body, "llrlrrrrl",
          "Out-of-sample check of the propagation factor and of the loss relation", "tab:extcheck", notes)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Write the LaTeX tables of the article from the result tables.")
    ap.add_argument("--results", help="results directory to read (default: paths.results of the config); "
                                      "'reference_results' regenerates the tables of the article without a GPU "
                                      "(on a copy in results/regenerated)")
    ap.add_argument("--out", help="output directory for the .tex tables (default: <results>/tables/tex)")
    args = ap.parse_args()
    cfg = load_config()
    if args.results:
        # Work on a copy: several tables also (re)write derived CSV files next to their inputs.
        import shutil
        work = CODE_ROOT / "results" / "regenerated"
        shutil.copytree((CODE_ROOT / args.results).resolve(), work, dirs_exist_ok=True)
        cfg["paths"]["results"] = work
        print(f"reading {args.results}, writing to {work}")
    out = Path(args.out).resolve() if args.out else results_dir(cfg, "tables", "tex")
    out.mkdir(parents=True, exist_ok=True)
    for f in (table_models, table_datasets, table_energy_reference, table_placement, table_accuracy_speed, table_layer_stats, table_aibo,
              table_hardware, table_engines, table_localization, table_propagation, table_selective,
              table_calibration_variability, table_operating_point, table_duplicates,
              table_calibration_methods, table_deployment, table_prediction, table_formulas, table_model_check, table_static,
              table_static_validation, table_static_percent, table_extended_check):
        f(cfg, out)
        print("done:", f.__name__, flush=True)
