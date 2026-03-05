from EchteAI.models.vision.models.fasterrcnn_utils import resize_and_pad
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, quantize_static, QuantType
import logging
import numpy as np
import torch

def quantize_onnx_model_calibdl(model_path, calib_data_loader, quantized_model_path, quantization_dtype="int16"):
    if quantization_dtype == "int8":
        weight_type = QuantType.QInt8
        activation_type = QuantType.QInt8
    else:
        weight_type = QuantType.QInt16
        activation_type = QuantType.QInt16

    quantized_model = quantize_static(
        model_input=model_path,
        model_output=quantized_model_path,
        weight_type=weight_type,
        activation_type=activation_type,
        calibration_data_reader=calib_data_loader
    )

    logging.info(f"Quantization ({quantization_dtype.upper()}) successful: {quantized_model_path}")    

def quantize_onnx_model_calibdl_int8(model_path, calib_data_loader, quantized_model_path):
    quantized_model = quantize_static(
    model_input=model_path,
    model_output=quantized_model_path,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QInt8,
    calibration_data_reader=calib_data_loader,
    quant_format="QDQ"
)
    
    logging.info(f"Quantization is successful: {quantized_model_path}")

class SingleImageCalibrationReader(CalibrationDataReader):
    def __init__(self, dataloader, input_name="images", target_size=(224,224), device="cpu"):
        self.dataloader = iter(dataloader)
        self.input_name = input_name
        self.target_size = target_size
        self.device = device

    def get_next(self):
        try:
            img, _ = next(self.dataloader)
            if isinstance(img, (list, tuple)):
                img = img[0]
            img = img.to(self.device)

            img_prepped = resize_and_pad(img, self.target_size)

            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3,1,1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3,1,1)
            img_prepped = (img_prepped - mean) / std

            img_prepped = img_prepped.unsqueeze(0)
            return {self.input_name: img_prepped.cpu().numpy().astype(np.float32)}
        except StopIteration:
            return None