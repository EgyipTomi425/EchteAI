"""Build TensorRT engines: static-shape engines for benchmarking, dynamic ones for evaluation."""
import argparse
import time

from pepai import debug
from pepai.config import load_config, results_dir
from pepai.models import SPECS
from pepai.trt import build_engine


def engine_jobs(cfg, models, precisions, batch_sizes=None):
    for name in models:
        spec = SPECS[name]
        for precision in precisions:
            onnx_path = results_dir(cfg, "onnx") / f"{name}_{precision}.onnx"
            for bs in batch_sizes or cfg["benchmark"]["batch_sizes"]:
                shape = (bs, *spec.bench_shape)
                yield name, precision, f"bs{bs}", onnx_path, {spec.input_name: (shape, shape, shape)}
            if spec.dynamic_hw and not batch_sizes:
                (hmin, hmax), (wmin, wmax) = spec.dynamic_hw
                c, h, w = spec.bench_shape
                shapes = ((1, c, hmin, wmin), (1, c, h, w), (1, c, hmax, wmax))
                if precision == "fp8":
                    # TensorRT 11.3 FP8 (Myelin) kernels fail at run time for inputs larger than the
                    # optimisation shape (20 of the 55 COCO input shapes of Faster R-CNN); optimising for
                    # the largest shape avoids this. Accuracy does not depend on the optimisation shape.
                    shapes = (shapes[0], shapes[2], shapes[2])
                yield name, precision, "dyn", onnx_path, {spec.input_name: shapes}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--precisions", nargs="*")
    ap.add_argument("--debug", action="store_true", help="build the analysis engines instead")
    ap.add_argument("--batch-sizes", nargs="*", type=int, help="static batch sizes instead of the configured ones")
    args = ap.parse_args()
    cfg = load_config()
    if args.debug:
        jobs = []
        for name in args.models or cfg["models"]:
            jobs += [(name, p, "full") for p in ("fp32", "int8fp32", "fp16")]
            if name in ("frcnn_r50_fpn", "yolov10s", "yolov10x"):
                jobs += [(name, p, "final") for p in ("fp32", "int8fp32")]
        for name, precision, kind in jobs:
            if not debug.engine_path(cfg, name, precision, kind).exists():
                t0 = time.time()
                path = debug.ensure_engine(cfg, name, precision, kind)
                print(f"{path.name}: built in {time.time() - t0:.0f} s", flush=True)
        raise SystemExit
    engine_dir = results_dir(cfg, "engines")
    cache = engine_dir / "timing.cache"
    for name, precision, tag, onnx_path, shapes in engine_jobs(
        cfg, args.models or cfg["models"], args.precisions or cfg["precisions"], args.batch_sizes
    ):
        out = engine_dir / f"{name}_{precision}_{tag}.engine"
        if out.exists():
            continue
        t0 = time.time()
        build_engine(onnx_path, out, shapes, timing_cache=cache)
        print(f"{out.name}: {out.stat().st_size / 1e6:.1f} MB, built in {time.time() - t0:.0f} s", flush=True)
