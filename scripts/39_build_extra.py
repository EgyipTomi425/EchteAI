"""Static-shape engines of the recommended DenseNet-121 INT8 variant (batch normalisation kept in FP16).

The variant comes from 17_placement_ablation.py (results/ablation/densenet121_bn_fp16_int8.onnx); the engines are
written next to the main engines as densenet121_int8bnfp16_bs{6,8}.engine so that 04_benchmark.py and
36_energy_graph.py can time them like every other precision.
"""
from pepai.config import load_config, results_dir
from pepai.models import SPECS
from pepai.trt import build_engine

if __name__ == "__main__":
    cfg = load_config()
    spec = SPECS["densenet121"]
    src = results_dir(cfg, "ablation") / "densenet121_bn_fp16_int8.onnx"
    for bs in (8, 6):
        out = results_dir(cfg, "engines") / f"densenet121_int8bnfp16_bs{bs}.engine"
        if not out.exists():
            build_engine(src, out, {spec.input_name: ((bs, *spec.bench_shape),) * 3},
                         timing_cache=results_dir(cfg, "engines") / "timing.cache")
            print("built", out.name, flush=True)
