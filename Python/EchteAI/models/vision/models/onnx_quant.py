from onnxruntime.quantization import quantize_static, QuantType
import logging

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
        calibration_data_reader=calib_data_loader
    )
    
    logging.info(f"Quantization is successful: {quantized_model_path}")