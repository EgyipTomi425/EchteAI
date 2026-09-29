"""Analysis ('debug') engines: TensorRT engines with intermediate tensors marked as outputs.

kind="full":  every analysed tensor (layer-wise activation analysis)
kind="final": only the head-input tensors (deviation-risk and robustness analysis)
Marking outputs changes layer fusion, so these engines are never used for timing.
"""
import onnx

from pepai import models
from pepai.activations import ANALYSED_OPS, analysed_tensors, final_tensors
from pepai.config import results_dir
from pepai.models import SPECS
from pepai.trt import build_engine


def engine_path(cfg, name, precision, kind):
    suffix = "" if kind == "full" else f"_{kind}"
    return results_dir(cfg, "engines", "debug") / f"{name}_{precision}{suffix}.engine"


def marked_tensors(cfg, name, precision, kind, reference_quant="int8fp32"):
    onnx_dir = results_dir(cfg, "onnx")
    fp32 = onnx_dir / f"{name}_fp32.onnx"
    if kind == "final":
        infos = analysed_tensors(fp32, onnx_dir / f"{name}_{reference_quant}.onnx")
        return final_tensors(cfg, name, fp32, infos)
    if precision == "fp32":
        # Superset: every analysed op output, so one reference engine serves all quantized variants.
        graph = onnx.load(str(fp32), load_external_data=False).graph
        return [o for n in graph.node if n.op_type in ANALYSED_OPS for o in n.output]
    return [t.name for t in analysed_tensors(fp32, onnx_dir / f"{name}_{precision}.onnx")]


def ensure_engine(cfg, name, precision, kind):
    path = engine_path(cfg, name, precision, kind)
    if not path.exists():
        build_engine(results_dir(cfg, "onnx") / f"{name}_{precision}.onnx", path,
                     models.engine_shapes(SPECS[name], opt_max=precision == "fp8"),
                     mark_outputs=marked_tensors(cfg, name, precision, kind))
    return path
