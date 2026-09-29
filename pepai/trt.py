"""TensorRT engine building and execution with PyTorch-owned device buffers."""
import os
from pathlib import Path

import tensorrt as trt
import torch

LOGGER = trt.Logger(trt.Logger.WARNING)

TRT_TO_TORCH = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT64: torch.int64,
    trt.DataType.INT8: torch.int8,
    trt.DataType.BOOL: torch.bool,
}


def build_engine(onnx_path, engine_path, input_shapes, mark_outputs=None, timing_cache=None,
                 workspace_gb=16):
    """Build a strongly typed engine: precision comes from the ONNX graph (QDQ / FP16 casts).

    input_shapes: {input_name: (min_shape, opt_shape, max_shape)}.
    mark_outputs: extra tensor names exposed as outputs (activation capture; changes fusions,
                  so such engines are never used for timing).
    TF32 is disabled so that the FP32 baseline is true FP32.
    """
    builder = trt.Builder(LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, LOGGER)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed for {onnx_path}:\n{errors}")

    if mark_outputs:
        wanted = set(mark_outputs)
        already = {network.get_output(i).name for i in range(network.num_outputs)}
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            for j in range(layer.num_outputs):
                t = layer.get_output(j)
                if t.name in wanted and t.name not in already:
                    network.mark_output(t)
                    already.add(t.name)

    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    # Per-layer metadata (formats, shapes) for the memory-traffic analysis; no effect on the kernels.
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    cache = None
    if timing_cache is not None:
        blob = Path(timing_cache).read_bytes() if Path(timing_cache).exists() else b""
        cache = config.create_timing_cache(blob) or config.create_timing_cache(b"")
        config.set_timing_cache(cache, ignore_mismatch=False)

    profile = builder.create_optimization_profile()
    for name, (mn, opt, mx) in input_shapes.items():
        profile.set_shape(name, mn, opt, mx)
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"Engine build failed for {onnx_path}")
    Path(engine_path).parent.mkdir(parents=True, exist_ok=True)
    Path(engine_path).write_bytes(bytes(serialized))
    if cache is not None:
        # Atomic replace: several build processes may share the cache.
        tmp = Path(timing_cache).with_suffix(f".{os.getpid()}.tmp")
        tmp.write_bytes(bytes(cache.serialize()))
        os.replace(tmp, timing_cache)
    return Path(engine_path)


class TRTModel:
    """Runs an engine on the current PyTorch CUDA stream. Inputs/outputs are torch CUDA tensors."""

    def __init__(self, engine_path):
        self.runtime = trt.Runtime(LOGGER)
        self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        self.context = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        self._out_buffers = {}

    def dtype(self, name):
        return TRT_TO_TORCH[self.engine.get_tensor_dtype(name)]

    def bind(self, **inputs):
        """Set shapes/addresses; returns output tensors (reused between calls with equal shapes)."""
        for name in self.inputs:
            x = inputs[name]
            assert x.is_cuda and x.is_contiguous() and x.dtype == self.dtype(name), name
            self.context.set_input_shape(name, tuple(x.shape))
            self.context.set_tensor_address(name, x.data_ptr())
        outs = {}
        for name in self.outputs:
            shape = tuple(self.context.get_tensor_shape(name))
            buf = self._out_buffers.get(name)
            if buf is None or tuple(buf.shape) != shape:
                buf = torch.empty(shape, dtype=self.dtype(name), device="cuda")
                self._out_buffers[name] = buf
            self.context.set_tensor_address(name, buf.data_ptr())
            outs[name] = buf
        return outs

    def execute(self):
        ok = self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT execution failed")

    def __call__(self, **inputs):
        outs = self.bind(**inputs)
        self.execute()
        return outs

    def device_memory_bytes(self):
        return self.engine.device_memory_size_v2
