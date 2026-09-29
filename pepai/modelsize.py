"""Deployed weight storage of an ONNX model (quantized weights count one byte)."""
import onnx

DTYPE_BYTES = {onnx.TensorProto.FLOAT: 4, onnx.TensorProto.FLOAT16: 2, onnx.TensorProto.BFLOAT16: 2,
               onnx.TensorProto.INT8: 1, onnx.TensorProto.UINT8: 1, onnx.TensorProto.FLOAT8E4M3FN: 1,
               onnx.TensorProto.INT32: 4, onnx.TensorProto.INT64: 8}


def weight_bytes(path):
    """Tensors feeding a QuantizeLinear/DequantizeLinear (INT8 or FP8 weights) count 1 byte each."""
    g = onnx.load(str(path)).graph
    quantized = {n.input[0] for n in g.node if n.op_type in ("QuantizeLinear", "DequantizeLinear")}
    total = 0
    for init in g.initializer:
        numel = 1
        for d in init.dims:
            numel *= d
        if numel <= 1:          # scales / zero points
            continue
        total += numel * (1 if init.name in quantized else DTYPE_BYTES.get(init.data_type, 4))
    return total
