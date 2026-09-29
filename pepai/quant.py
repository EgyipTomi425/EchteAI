"""FP16 conversion and INT8 post-training quantization (explicit QDQ) with NVIDIA ModelOpt."""
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import onnx
import torch
from onnxruntime.quantization import CalibrationDataReader

from pepai import data, models


class ListReader(CalibrationDataReader):
    """Feeds calibration samples one by one; needed when spatial shapes vary (Faster R-CNN)."""

    def __init__(self, input_name, arrays):
        self.input_name = input_name
        self.arrays = arrays
        self.it = iter(arrays)

    def get_next(self):
        x = next(self.it, None)
        return None if x is None else {self.input_name: x}

    def rewind(self):
        self.it = iter(self.arrays)


def calibration_inputs(cfg, name):
    """Preprocessed calibration inputs: a stacked array, or a list of arrays for dynamic shapes."""
    if name == "frcnn_r50_fpn":
        frcnn = models.load_frcnn().cuda()
        out = []
        for p in data.coco_calib_paths(cfg):
            x = models.frcnn_preprocess(frcnn, [data.load_rgb(p)]).tensors
            out.append(x.cpu().numpy().astype(np.float32))
        return out
    if name.startswith("yolov10"):
        xs = [models.letterbox(data.load_rgb(p))[0] for p in data.coco_calib_paths(cfg)]
        return torch.stack(xs).numpy().astype(np.float32)
    calib, _ = data.imagenetv2_split(cfg)
    xs = [models.classifier_preprocess(data.load_rgb(p)) for p, _ in calib]
    return torch.stack(xs).numpy().astype(np.float32)


@contextmanager
def scratch_copy(fp32_path):
    """ModelOpt rewrites its input and drops intermediates next to it; keep the FP32 reference pristine."""
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / Path(fp32_path).name
        shutil.copy(fp32_path, copy)
        yield copy


def autocast_sample(calib, n=16):
    """A single real batch for ModelOpt autocast, which keeps nodes whose reference activations
    exceed the FP16-safe range in FP32. Variable-size inputs are zero-padded to a common shape."""
    if not isinstance(calib, list):
        return calib[:n]
    xs = calib[:n]
    h = max(x.shape[2] for x in xs)
    w = max(x.shape[3] for x in xs)
    out = np.zeros((len(xs), xs[0].shape[1], h, w), dtype=np.float32)
    for i, x in enumerate(xs):
        out[i, :, :x.shape[2], :x.shape[3]] = x[0]
    return out


def to_fp16(fp32_path, out_path, sample):
    from modelopt.onnx.autocast import convert_to_mixed_precision
    with scratch_copy(fp32_path) as src:
        input_name = onnx.load(str(src), load_external_data=False).graph.input[0].name
        npz = src.with_suffix(".npz")
        np.savez(npz, **{input_name: sample})
        model = convert_to_mixed_precision(
            onnx_path=str(src), low_precision_type="fp16", keep_io_types=True,
            calibration_data=str(npz), providers=["cuda:0", "cpu"],
        )
    onnx.save(model, str(out_path))


def to_int8(fp32_path, out_path, calib, high_precision_dtype="fp32", method="entropy",
            nodes_to_exclude=None, op_types_to_exclude=None):
    """INT8 QDQ: symmetric per-channel weights, symmetric per-tensor activations (TensorRT convention).

    nodes_to_exclude: FP32-graph node names kept in high precision (selective quantization).
    """
    from modelopt.onnx.quantization import quantize
    with scratch_copy(fp32_path) as src:
        kwargs = dict(
            onnx_path=str(src), quantize_mode="int8", calibration_method=method,
            high_precision_dtype=high_precision_dtype, output_path=str(out_path),
            calibration_eps=["cuda:0", "cpu"], log_level="WARNING",
            nodes_to_exclude=nodes_to_exclude or None,
            op_types_to_exclude=op_types_to_exclude or None,
        )
        if isinstance(calib, list):
            input_name = onnx.load(str(src), load_external_data=False).graph.input[0].name
            kwargs["calibration_data_reader"] = ListReader(input_name, calib)
        else:
            kwargs["calibration_data"] = calib
        quantize(**kwargs)
    # ModelOpt leaves intermediate graphs next to the output file.
    for suffix in ("_opset19.onnx", "_reconciled.onnx", "_named.onnx"):
        Path(out_path).with_name(Path(fp32_path).stem + suffix).unlink(missing_ok=True)
    return Path(out_path)


def int8_with_fp16(int8fp32_path, out_path):
    """Same QDQ scales, non-quantized ops in FP16 (the TensorRT deployment setting).

    ModelOpt applies the high-precision dtype only after calibration, so converting the
    INT8+FP32 model reproduces quantize(high_precision_dtype="fp16") without calibrating twice.
    """
    from modelopt.onnx.quantization.precision_utils import _convert_to_runtime_precision
    model = onnx.load(str(int8fp32_path))
    model = _convert_to_runtime_precision(model, quantize_mode="int8", high_precision_dtype="fp16")
    onnx.save(model, str(out_path))


QUANTIZED_COMPUTE = {"Conv", "ConvTranspose", "Gemm", "MatMul"}
EPILOGUE = {"Sigmoid", "Mul", "Relu", "Clip", "HardSigmoid", "HardSwish"}


def release_output_qdq(path, excluded_nodes):
    """Drop the Q/DQ pair on the activation output of every excluded node, unless it feeds a
    quantized Conv/Gemm/MatMul. Keeps the excluded layer's epilogue in high precision and avoids
    FP16-compute/INT8-output fusions that TensorRT has no kernel for (e.g. depthwise conv + SiLU)."""
    import onnx_graphsurgeon as gs
    graph = gs.import_onnx(onnx.load(str(path)))
    by_name = {n.name: n for n in graph.nodes}
    released = 0
    for name in excluded_nodes:
        # Tensors of the excluded node's epilogue chain (e.g. Conv -> Sigmoid -> Mul for SiLU).
        tensors, frontier = [], list(by_name[name].outputs)
        while frontier:
            t = frontier.pop()
            if any(t is u for u in tensors):     # SiLU reaches its Mul twice
                continue
            tensors.append(t)
            for c in list(t.outputs):
                if c.op in EPILOGUE:
                    frontier.extend(c.outputs)
        for t in tensors:
            for q in [c for c in list(t.outputs) if c.op == "QuantizeLinear" and c.outputs]:
                dqs = [c for c in q.outputs[0].outputs if c.op == "DequantizeLinear"]
                consumers = [c for d in dqs for c in d.outputs[0].outputs]
                if any(c.op in QUANTIZED_COMPUTE for c in consumers):
                    continue
                for d in dqs:
                    for c in list(d.outputs[0].outputs):
                        c.inputs = [t if i is d.outputs[0] else i for i in c.inputs]
                    d.outputs.clear()
                q.outputs.clear()
                released += 1
    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), str(path))
    return released


def _kernel_closure(path):
    """Nodes that must also run in FP16 because TensorRT has no kernel for the precision mix around a
    depthwise convolution: (a) a quantized depthwise convolution whose output (through its epilogue)
    reaches no QuantizeLinear, i.e. INT8 input with a floating-point output; (b) the quantized consumers
    of an FP16 depthwise convolution whose epilogue output is quantized directly, i.e. FP16 input with an
    INT8 output fused into the convolution."""
    import onnx_graphsurgeon as gs
    graph = gs.import_onnx(onnx.load(str(path)))
    out = []
    for n in graph.nodes:
        if n.op != "Conv" or int(n.attrs.get("group", 1)) == 1:
            continue
        quantized_in = bool(n.inputs[0].inputs) and n.inputs[0].inputs[0].op == "DequantizeLinear"
        tensors, frontier, qs = [], list(n.outputs), []
        while frontier:
            t = frontier.pop()
            if any(t is u for u in tensors):
                continue
            tensors.append(t)
            for c in t.outputs:
                if c.op == "QuantizeLinear":
                    qs.append(c)
                elif c.op in EPILOGUE:
                    frontier.extend(c.outputs)
        if quantized_in and not qs:
            out.append(n.name)
        elif not quantized_in and qs:
            for q in qs:
                for dq in [c for c in q.outputs[0].outputs if c.op == "DequantizeLinear"]:
                    out += [c.name for c in dq.outputs[0].outputs if c.op in QUANTIZED_COMPUTE]
    return list(dict.fromkeys(out))


def exclude_from_quantized(src_path, out_path, excluded_nodes):
    """Selective precision without re-calibration: rewire the activation and weight inputs of every
    excluded node from its Dequantize output to the original floating-point tensor, then release its
    output quantizers (release_output_qdq). All remaining quantizers keep the scales calibrated for
    the fully quantized model, so variants differ only in the excluded nodes. Q/DQ pairs still used
    by other consumers stay in place.

    Closure: nodes for which TensorRT would have no kernel after the exclusion are excluded as well
    (see _kernel_closure). Returns the effective excluded list."""
    import onnx_graphsurgeon as gs
    excluded = list(excluded_nodes)
    while True:
        graph = gs.import_onnx(onnx.load(str(src_path)))
        by_name = {n.name: n for n in graph.nodes}
        missing = [n for n in excluded if n not in by_name]
        if missing:
            raise KeyError(f"not in the quantized graph: {missing}")
        for name in excluded:
            node = by_name[name]
            inputs = list(node.inputs)
            for k, t in enumerate(inputs):
                dq = t.inputs[0] if t.inputs else None
                if dq is None or dq.op != "DequantizeLinear" or not dq.inputs[0].inputs:
                    continue
                q = dq.inputs[0].inputs[0]
                if q.op == "QuantizeLinear":
                    inputs[k] = q.inputs[0]
            node.inputs = inputs
        graph.cleanup().toposort()
        onnx.save(gs.export_onnx(graph), str(out_path))
        release_output_qdq(out_path, excluded)
        extra = [n for n in _kernel_closure(out_path) if n not in excluded]
        if not extra:
            return excluded
        excluded += extra


def to_fp8(fp32_path, out_path, calib, high_precision_dtype="fp16", op_types_to_exclude=None,
           nodes_to_exclude=None):
    """FP8 (E4M3) QDQ for Hopper tensor cores; same calibration data as INT8."""
    from modelopt.onnx.quantization import quantize
    with scratch_copy(fp32_path) as src:
        kwargs = dict(onnx_path=str(src), quantize_mode="fp8", high_precision_dtype=high_precision_dtype,
                      output_path=str(out_path), calibration_eps=["cuda:0", "cpu"], log_level="WARNING",
                      op_types_to_exclude=op_types_to_exclude or None, nodes_to_exclude=nodes_to_exclude or None)
        if isinstance(calib, list):
            input_name = onnx.load(str(src), load_external_data=False).graph.input[0].name
            kwargs["calibration_data_reader"] = ListReader(input_name, calib)
        else:
            kwargs["calibration_data"] = calib
        quantize(**kwargs)
    for suffix in ("_opset19.onnx", "_reconciled.onnx", "_named.onnx"):
        Path(out_path).with_name(Path(fp32_path).stem + suffix).unlink(missing_ok=True)
    return Path(out_path)


def excluded_op_types(cfg, name):
    return cfg.get("quantization", {}).get("exclude_op_types", {}).get(name, [])


def excluded_nodes(cfg, name):
    return cfg.get("quantization", {}).get("exclude_nodes", {}).get(name, [])
