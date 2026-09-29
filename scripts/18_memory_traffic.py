"""Memory traffic and kernel precision per inference from the TensorRT engine inspector (CPU only).

Hardware counters (Nsight Compute) need administrator rights on this system, so traffic is computed
from the deployed engine itself: for every fused layer, the bytes of its input and output tensors
(in their actual data types) plus its weights and bias. Intermediate tensors are therefore counted
once when written and once when read, which is the DRAM traffic without cross-layer caching.
The kernel (tactic) names show whether a layer runs on integer (IMMA), half-precision (HMMA) or
FP32 units, i.e. whether INT8 is really executed as INT8.

Needs engines built with ProfilingVerbosity.DETAILED (pepai.trt.build_engine does this).
"""
import argparse
import json
import re

import numpy as np
import pandas as pd
import tensorrt as trt

from pepai.config import load_config, results_dir
from pepai.modelsize import weight_bytes
from pepai.trt import LOGGER

BYTES = {"Int8": 1, "FP8": 1, "UInt8": 1, "Bool": 1, "Half": 2, "BFloat16": 2, "Float": 4, "Int32": 4,
         "Int64": 8}


def tensor_bytes(t):
    dims = t.get("Dimensions", [])
    if not dims or any(d < 0 for d in dims):
        return 0
    return int(np.prod(dims)) * BYTES.get(t.get("Datatype", "Float"), 4)


def unit(layer):
    """Arithmetic precision of a convolution-like layer from its kernel name and input data type."""
    t = layer.get("TacticName", "").lower()
    dtype = ((layer.get("Inputs") or [{}])[0].get("Datatype") or "").lower()
    if "e4m3" in t or "e5m2" in t or dtype == "fp8":
        return "fp8"
    if "imma" in t or "i8i8" in t or dtype == "int8":
        return "int8"
    if "f16" in t or "hmma" in t or dtype == "half":
        return "fp16"
    return "other"


def analyse(engine_path):
    runtime = trt.Runtime(LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    info = json.loads(engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
    act = wts = 0
    units = {"int8": 0, "fp8": 0, "fp16": 0, "other": 0}
    conv_like = 0
    for layer in info["Layers"]:
        if not isinstance(layer, dict):
            raise ValueError(f"{engine_path.name}: built without detailed profiling verbosity")
        act += sum(tensor_bytes(t) for t in layer.get("Inputs", []) + layer.get("Outputs", []))
        for key in ("Weights", "Bias"):
            w = layer.get(key)
            if isinstance(w, dict):
                wts += w.get("Count", 0) * BYTES.get(w.get("Type", "Float"), 4)
        # Convolutions appear as Cask* layers or, inside Myelin regions, as "correlation" layers.
        if layer.get("LayerType", "") in ("CaskConvolution", "CaskGemmConvolution", "CaskDeconvolution",
                                          "correlation", "gemm", "CaskGemm") or re.search("Conv", layer.get("LayerType", "")):
            conv_like += 1
            units[unit(layer)] += 1
    types = [layer.get("LayerType", "") for layer in info["Layers"]]
    kernels = sum(t not in ("NoOp", "wait", "signal") for t in types)     # launched layers (fused kernels)
    return {"layers": len(info["Layers"]), "kernels": kernels, "reformats": types.count("Reformat"),
            "pointwise_separate": types.count("kgen") + types.count("PointWiseV2") + types.count("PointWise"),
            "conv_layers": conv_like, "activation_mb": act / 1e6,
            "weight_mb": wts / 1e6, "traffic_mb": (act + wts) / 1e6,
            **{f"conv_on_{k}": v for k, v in units.items()}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine-dir", help="default: results/engines")
    args = ap.parse_args()
    cfg = load_config()
    d = results_dir(cfg, "engines") if not args.engine_dir else __import__("pathlib").Path(args.engine_dir)
    rows = []
    for path in sorted(d.glob("*_bs1.engine")):
        name, precision = re.match(r"(.+)_(fp32|fp16|int8fp32|int8|fp8)_bs1\.engine", path.name).groups()
        try:
            rows.append({"model": name, "precision": precision, **analyse(path)})
        except ValueError as e:
            print("skipped:", e)
    df = pd.DataFrame(rows)
    if not df.empty:
        # Myelin regions do not report weights; take the deployed weight bytes from the ONNX model.
        onnx_dir = results_dir(cfg, "onnx")
        df["weight_mb"] = [weight_bytes(onnx_dir / f"{r.model}_{r.precision}.onnx") / 1e6 for r in df.itertuples()]
        df["traffic_mb"] = df.activation_mb + df.weight_mb
        ref = df[df.precision == "fp32"].set_index("model").traffic_mb
        df["traffic_vs_fp32"] = df.apply(lambda r: r.traffic_mb / ref.get(r.model, np.nan), axis=1)
        df.to_csv(results_dir(cfg, "tables") / "memory_traffic.csv", index=False)
        print(df.round(2).to_string(index=False))
