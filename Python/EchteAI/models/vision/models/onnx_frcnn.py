import onnx
import onnxruntime as ort
import torch
import numpy as np
import re
import os
import torch.nn.functional as F
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
import logging

class FeatureExtractorCalibrationDataReader(CalibrationDataReader):
    def __init__(self, data_loader, input_name, input_shape, num_batches):
        self.input_name = input_name
        self.input_shape = input_shape
        self.inputs = []

        bs, c, h, w = input_shape
        count = 0
        for images, _ in data_loader:
            if count >= num_batches:
                break
            if len(images) < bs:
                continue 
            batch = torch.stack([
                F.interpolate(img.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze(0)
                for img in images[:bs]
            ])
            self.inputs.append({input_name: batch.numpy()})
            count += 1

        self.input_iter = iter(self.inputs)

    def get_next(self):
        return next(self.input_iter, None)
    
def quantize_onnx_static(onnx_model_path, data_loader, input_shape=(2, 3, 375, 1242), num_batches=8):
    model = onnx.load(onnx_model_path)
    input_name = model.graph.input[0].name
    base, ext = os.path.splitext(onnx_model_path)
    quantized_model_path = base + "_int8" + ext
    dr = FeatureExtractorCalibrationDataReader(data_loader, input_name, input_shape, num_batches)
    quantize_static(
        model_input=onnx_model_path,
        model_output=quantized_model_path,
        calibration_data_reader=dr,
        quant_format="QDQ",
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
    )
    logging.info(f"Quantized model saved successfully: {quantized_model_path}")

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