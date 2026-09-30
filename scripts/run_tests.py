"""Sequential test scheduler: runs exactly one measurement at a time, as soon as its inputs exist.

Builds and quantization may run in parallel elsewhere; this process never starts a second test
while one is running. Completed tasks are recorded in results/logs/done.txt.
"""
import os
import subprocess
import sys
import time

from pepai import debug
from pepai.config import CODE_ROOT, load_config, results_dir
from pepai.models import SPECS

DETECTORS = ["frcnn_r50_fpn", "yolov10s", "yolov10x"]
ORDER = ["efficientnet_b0", "densenet121", "yolov10s", "yolov10x", "frcnn_r50_fpn"]


def std_engines(cfg, name, precisions=None):
    d = results_dir(cfg, "engines")
    tags = [f"bs{b}" for b in cfg["benchmark"]["batch_sizes"]] + (["dyn"] if SPECS[name].dynamic_hw else [])
    return [d / f"{name}_{p}_{t}.engine" for p in (precisions or cfg["precisions"]) for t in tags]


def tasks(cfg):
    py = sys.executable
    out = []
    # Early go/no-go on speed (latency only); the clean benchmark runs last (final_benchmark).
    out.append(("speed_check", [py, "scripts/04_benchmark.py", "--quick", "--out", "benchmark_quick.csv",
                                "--precisions", "fp32", "fp16", "int8", "fp8"],
                [p for m in ORDER for p in std_engines(cfg, m, ["fp32", "fp16", "int8", "fp8"])]))
    for m in ORDER:
        out.append((f"accuracy_{m}", [py, "scripts/05_accuracy.py", "--models", m], std_engines(cfg, m)))
    for m in ORDER:
        for q in ("int8fp32", "fp16"):
            # FP16 is a lossless control (42-50 dB on four models); 100 images suffice for the largest model.
            n_img = ["--n", "100"] if (q == "fp16" and m == "frcnn_r50_fpn") else []
            out.append((f"activations_{m}_{q}", [py, "scripts/06_activations.py", "--models", m, "--quant", q, *n_img],
                        [debug.engine_path(cfg, m, "fp32", "full"), debug.engine_path(cfg, m, q, "full")]))
    out.append(("placement_densenet121", [py, "scripts/17_placement_ablation.py", "--models", "densenet121"],
                std_engines(cfg, "densenet121") + [debug.engine_path(cfg, "densenet121", "fp32", "full")]))
    for m in ("yolov10s", "yolov10x", "frcnn_r50_fpn"):
        out.append((f"placement_{m}", [py, "scripts/17_placement_ablation.py", "--models", m],
                    std_engines(cfg, m) + [debug.engine_path(cfg, m, "fp32", "final")]))
    abl = results_dir(cfg, "ablation")
    out.append(("label_free_selection", [py, "scripts/20_label_free_selection.py"],
                [abl / "densenet121_bn_fp16_calib_max_int8_bs8.engine", abl / "yolov10s_calib_max_int8_bs1.engine",
                 abl / "yolov10x_calib_max_int8_bs1.engine", abl / "frcnn_r50_fpn_calib_max_int8_bs1.engine"]))
    for m in DETECTORS:
        final = [debug.engine_path(cfg, m, p, "final") for p in ("fp32", "int8fp32")]
        out.append((f"risk_{m}", [py, "scripts/07_risk.py", "--models", m], final))
        out.append((f"robustness_{m}", [py, "scripts/08_robustness.py", "--models", m],
                    final + std_engines(cfg, m)))
    for m in ("yolov10s", "yolov10x"):
        out.append((f"robust_calibration_{m}", [py, "scripts/21_robust_calibration.py", "--models", m],
                    [cfg["coco"]["root"] / "coco_c" / c / ".complete" for c in ("contrast", "fog", "dark", "motion_blur")]))
    coco_c = cfg["coco"]["root"] / "coco_c"
    for cond in ("fog", "dark"):
        for sev in (1, 2, 3, 4, 5):
            out.append((f"surface_frcnn_r50_fpn_{cond}{sev}",
                        [py, "scripts/06_activations.py", "--models", "frcnn_r50_fpn", "--quant", "int8fp32",
                         "--n", "100", "--condition", cond, "--severity", str(sev)],
                        [debug.engine_path(cfg, "frcnn_r50_fpn", p, "full") for p in ("fp32", "int8fp32")]
                        + [coco_c / cond / ".complete"]))
    act = results_dir(cfg, "activations")
    out.append(("selective_efficientnet_b0", [py, "scripts/10_selective.py", "--models", "efficientnet_b0"],
                [act / "efficientnet_b0_int8fp32.csv.gz"]))
    for m in ("yolov10s", "yolov10x", "densenet121", "frcnn_r50_fpn"):
        acc = results_dir(cfg, "tables") / "accuracy.csv"
        limit = ["--limit", "2000"] if m == "frcnn_r50_fpn" else []
        out.append((f"selective_{m}", [py, "scripts/10_selective.py", "--models", m, "--strategies", "pepai",
                                       "--ks", "0", "1", "2", "3", "5", "8", "12", "20", *limit],
                    [act / f"{m}_int8fp32.csv.gz", act / f"{m}_int8fp32_tensors.json", acc]))
    # YOLOv10-X recovers most of its loss between k = 12 and 20: extend the curve.
    out.append(("selective_yolov10x_more", [py, "scripts/10_selective.py", "--models", "yolov10x", "--strategies",
                                            "pepai", "--ks", "30", "40"], [act / "yolov10x_int8fp32.csv.gz"]))
    # Theory-driven ranking (FP32 statistics only) versus the measured ranking; k = 0 is shared.
    # (Variants that failed to build before the kernel-closure rule are regenerated on re-run.)
    for m in ("yolov10s", "efficientnet_b0"):
        out.append((f"selective_noise_{m}", [py, "scripts/10_selective.py", "--models", m, "--strategies", "noise",
                                             "--ks", "1", "2", "3", "5", "8", "12", "20"],
                    [results_dir(cfg, "tables") / "quantizer_snr.csv"]))
    out.append(("localization", [py, "scripts/16_localization.py"],
                [results_dir(cfg, "risk") / f"{m}_int8fp32.csv" for m in DETECTORS]))
    for m in ORDER:   # Hopper FP8 extension (Faster R-CNN FP8 dyn engine: optimisation shape = max, see 03)
        out.append((f"accuracy_fp8_{m}", [py, "scripts/05_accuracy.py", "--models", m, "--precisions", "fp8"],
                     std_engines(cfg, m, ["fp8"])))
    # Noise-model prediction: FP8 keeps a constant relative precision, so unlike static per-tensor INT8
    # its penalty should not grow when image contrast (and hence the activation RMS) drops.
    out.append(("robustness_fp8", [py, "scripts/08_robustness.py", "--models", "yolov10s", "yolov10x",
                                   "--conditions", "contrast", "fog", "--precisions", "fp8"],
                [results_dir(cfg, "engines") / f"{m}_fp8_bs1.engine" for m in ("yolov10s", "yolov10x")]))
    # Layer-wise test of the static-scale prediction (Eq. static): SQNR drop of INT8 vs FP8 under contrast
    # reduction c = 0.4 / 0.2 / 0.05 (severities 1 / 3 / 5), same 100 images as the clean analysis.
    for m in ORDER:     # FP8 layer profile on clean images (precision spectrum of the propagation figure)
        out.append((f"layers_fp8_clean_{m}", [py, "scripts/06_activations.py", "--models", m, "--quant", "fp8",
                                             "--n", "100"], [debug.engine_path(cfg, m, "fp32", "full")]))
    for m in ("yolov10s", "yolov10x"):
        ref = [debug.engine_path(cfg, m, "fp32", "full"), coco_c / "contrast" / ".complete"]
        for sev in (1, 3, 5):
            out.append((f"layers_contrast_{m}_{sev}",
                        [py, "scripts/06_activations.py", "--models", m, "--quant", "int8fp32", "fp8", "--n", "100",
                         "--condition", "contrast", "--severity", str(sev)], ref))
    # Clipping mechanism under low contrast: entropy vs max calibration (ablation engines).
    abl = results_dir(cfg, "ablation")
    out.append(("calibration_robustness", [py, "scripts/27_calibration_robustness.py"],
                [abl / f"{m}_{v}_int8_bs1.engine" for m in ("yolov10s", "yolov10x") for v in ("default", "calib_max")]))
    out.append(("qualitative", [py, "scripts/28_qualitative.py"],
                [debug.engine_path(cfg, "yolov10x", "fp32", "final"), coco_c / "contrast" / ".complete"]))
    det = results_dir(cfg, "detections")
    out.append(("bootstrap_ci", [py, "scripts/22_bootstrap_ci.py"],
                [det / f"{m}_fp8.json" for m in DETECTORS]
                + [det / f"{m}_{p}_cls.npz" for m in ("efficientnet_b0", "densenet121") for p in ("fp32", "int8", "fp8")]))
    # Revision runs: batch size of the fleet scenario (six cameras) and calibration-sample sensitivity.
    out.append(("benchmark_bs6", ["bash", "-c", f"{py} scripts/03_build_engines.py --batch-sizes 6 --precisions fp32 fp16 "
                                  f"int8 fp8 && {py} scripts/04_benchmark.py --batch-sizes 6 --precisions fp32 fp16 int8 "
                                  f"fp8 --out benchmark_bs6.csv"], []))
    out.append(("calibration_variability", [py, "scripts/29_calibration_variability.py"], []))
    out.append(("calibration_size_classifiers", ["bash", "-c", f"{py} scripts/29_calibration_variability.py --models "
                                  f"efficientnet_b0 densenet121 --fraction 1.0 --seeds 1 && {py} scripts/29_calibration_variability.py "
                                  f"--models efficientnet_b0 densenet121 --fraction 0.25"], []))
    out.append(("global_entropy", [py, "scripts/34_global_entropy.py"], []))
    out.append(("conv_only_placement", [py, "scripts/35_conv_only_placement.py"],
                [results_dir(cfg, "calib_global") / "efficientnet_b0_global_thresholds.json"]))
    out.append(("selective_duplicates", [py, "scripts/38_selective_duplicates.py", "--ks", "20", "12"], []))
    out.append(("nms_cost", [py, "scripts/37_nms_cost.py"], []))
    out.append(("energy_graph", [py, "scripts/36_energy_graph.py"], []))
    out.append(("usability_extra", ["bash", "-c", f"{py} scripts/39_build_extra.py && {py} scripts/04_benchmark.py "
                                    f"--models densenet121 --precisions int8bnfp16 --batch-sizes 8 6 --out benchmark_extra.csv"
                                    f" && {py} scripts/36_energy_graph.py --models densenet121 --precisions int8bnfp16"], []))
    out.append(("scale_swap_efficientnet", [py, "scripts/31_scale_swap.py", "--model", "efficientnet_b0", "--seed", "1"],
                [results_dir(cfg, "calib_variability") / "efficientnet_b0_s1_int8fp32.onnx"]))
    # Runs last, only when every other task is done (see the main loop): clean rebuild + benchmark.
    out.append(("final_benchmark", ["env", f"PY={py}", "bash", "scripts/final_benchmark.sh"], []))
    out.append(("model_stats", [py, "scripts/09_model_stats.py"],
                [results_dir(cfg, "onnx") / f"{m}_int8.onnx" for m in ORDER]))
    return out


TEST_SCRIPTS = ("04_benchmark", "05_accuracy", "06_activations", "07_risk", "08_robustness",
                "09_model_stats", "10_selective", "16_localization", "17_placement_ablation",
                "20_label_free_selection", "21_robust_calibration", "22_bootstrap_ci",
                "final_benchmark", "29_calibration_variability", "31_scale_swap", "34_global_entropy", "35_conv_only_placement",
                "36_energy_graph", "37_nms_cost", "38_selective_duplicates")


def other_test_running():
    """True while any measurement script runs (e.g. one started by a previous scheduler instance)."""
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    return any(f"scripts/{t}.py" in line and "python" in line for line in out.splitlines() for t in TEST_SCRIPTS)


if __name__ == "__main__":
    logs = results_dir(load_config(), "logs")
    done_file = logs / "done.txt"
    attempted = set()
    started_mtime = os.path.getmtime(__file__)
    while True:
        # Task definitions are code: pick up edits of this file between tasks by re-executing it.
        if os.path.getmtime(__file__) != started_mtime:
            print(time.strftime("%H:%M"), "task list changed, restarting scheduler", flush=True)
            os.execv(sys.executable, [sys.executable] + sys.argv)
        cfg = load_config()
        done = set(done_file.read_text().split()) if done_file.exists() else set()
        pending = [t for t in tasks(cfg) if t[0] not in done and t[0] not in attempted]
        if not pending:
            break
        others = [t for t in pending if not t[0].startswith("final_")]
        ready = [t for t in pending if all(p.exists() for p in t[2])
                 and (not t[0].startswith("final_") or not others)]
        if not ready or other_test_running():
            time.sleep(60)
            continue
        name, cmd, _ = ready[0]
        print(time.strftime("%H:%M"), "start", name, flush=True)
        with open(logs / f"{name}.log", "w") as log:
            rc = subprocess.run(cmd, cwd=CODE_ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        print(time.strftime("%H:%M"), "end", name, "rc", rc, flush=True)
        if rc == 0:
            # Completed tasks are tracked in done.txt only, so deleting a line there re-runs the task.
            with open(done_file, "a") as f:
                f.write(name + "\n")
        else:
            attempted.add(name)
    print("TEST_QUEUE_DONE", flush=True)
