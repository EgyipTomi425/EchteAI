"""Create the FP16 and INT8 variants of every FP32 ONNX model."""
import argparse

from pepai.config import load_config, results_dir
from pepai.quant import (autocast_sample, calibration_inputs, excluded_nodes, excluded_op_types, int8_with_fp16,
                         to_fp16, to_fp8, to_int8)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--fp8", action="store_true", help="only create the FP8 variants (Hopper extension)")
    args = ap.parse_args()

    cfg = load_config()
    onnx_dir = results_dir(cfg, "onnx")
    for name in args.models or cfg["models"]:
        fp32 = onnx_dir / f"{name}_fp32.onnx"
        if args.fp8:
            out = onnx_dir / f"{name}_fp8.onnx"
            if not out.exists():
                to_fp8(fp32, out, calibration_inputs(cfg, name), op_types_to_exclude=excluded_op_types(cfg, name),
                       nodes_to_exclude=excluded_nodes(cfg, name))
            print(f"{name} fp8: {out.stat().st_size / 1e6:.1f} MB", flush=True)
            continue
        targets = {
            "fp16": onnx_dir / f"{name}_fp16.onnx",
            "int8": onnx_dir / f"{name}_int8.onnx",
            "int8fp32": onnx_dir / f"{name}_int8fp32.onnx",
        }
        if not (targets["fp16"].exists() and targets["int8fp32"].exists()):
            calib = calibration_inputs(cfg, name)
        if not targets["fp16"].exists():
            to_fp16(fp32, targets["fp16"], autocast_sample(calib))
        if not targets["int8fp32"].exists():
            to_int8(fp32, targets["int8fp32"], calib, high_precision_dtype="fp32",
                    op_types_to_exclude=excluded_op_types(cfg, name), nodes_to_exclude=excluded_nodes(cfg, name))
        if not targets["int8"].exists():
            int8_with_fp16(targets["int8fp32"], targets["int8"])
        for precision, path in targets.items():
            print(f"{name} {precision}: {path.stat().st_size / 1e6:.1f} MB")
