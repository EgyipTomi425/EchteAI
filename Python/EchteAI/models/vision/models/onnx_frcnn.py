import onnx
import onnxruntime as ort
import torch
import numpy as np
import re
import os
import torch.nn.functional as F
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
import logging

class FECalibrationDataReader(CalibrationDataReader):
    def __init__(self, data_loader, transform, input_name, num_batches=8):
        self.input_name = input_name
        self.inputs = []

        count = 0
        for images, _ in data_loader:
            if count >= num_batches:
                break

            img_list, _ = transform(images)
            tensors = img_list.tensors

            self.inputs.append({input_name: tensors.cpu().numpy()})
            count += 1

        self.iterator = iter(self.inputs)

    def get_next(self):
        return next(self.iterator, None)
    
def quantize_feature_extractor(fe_onnx_path, data_loader, transform, output_path, num_batches=8):
    model = onnx.load(fe_onnx_path)
    input_name = model.graph.input[0].name

    dr = FECalibrationDataReader(data_loader, transform, input_name, num_batches=num_batches)

    quantize_static(
        model_input=fe_onnx_path,
        model_output=output_path,
        calibration_data_reader=dr,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        quant_format="QDQ",
    )
    print(f"FE quantized model saved to: {output_path}")

def onnx_conv_outputs_from_batch(model_path, input_tensor, pattern=r".*conv.*"):
    model = onnx.load(model_path)
    conv_outputs = []
    for node in model.graph.node:
        if node.op_type.lower() == "conv":
            for output in node.output:
                if re.search(pattern, output, re.IGNORECASE):
                    conv_outputs.append(output)
    existing_outputs = [o.name for o in model.graph.output]
    value_infos = {vi.name: vi for vi in model.graph.value_info}
    for name in conv_outputs:
        if name not in existing_outputs and name in value_infos:
            model.graph.output.append(value_infos[name])
    export_path = model_path.replace(".onnx", "_with_outputs.onnx")
    onnx.save(model, export_path)
    session = ort.InferenceSession(export_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    ort_outputs = session.run(None, {input_name: input_tensor.cpu().numpy()})
    output_names = [o.name for o in session.get_outputs()]
    return {name: torch.tensor(val) for name, val in zip(output_names, ort_outputs)}