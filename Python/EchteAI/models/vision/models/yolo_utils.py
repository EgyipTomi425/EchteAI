from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
import os
import cv2
import numpy as np
import torch
import logging
from glob import glob
from onnxruntime.quantization import CalibrationDataReader
import onnx
import onnxruntime as ort

def setup_yolo(model_name="yolo11x.pt", pretrained=True):
    if os.path.exists("./self_"+model_name):
        model = YOLO("./self_"+model_name)
    else:
        model = YOLO(model_name)
        if not pretrained:
            model = model.reset()
    return model

def train_yolo(model, data_yaml_path, device, epochs=10, model_name="yolo11x.pt"):
    if os.path.exists("./self_"+model_name):
        return model
    model.train(data=data_yaml_path, epochs=epochs, device=device, weight_decay=0.001)
    model.save("self_"+model_name)
    return model

def compute_metrics_yolo(model, data_yaml_path, device):
    metrics = model.val(data=data_yaml_path, device=device)
    return {
        "precision": float(metrics.results_dict["metrics/precision(B)"]),
        "recall": float(metrics.results_dict["metrics/recall(B)"]),
        "mAP50": float(metrics.results_dict["metrics/mAP50(B)"]),
        "mAP50-95": float(metrics.results_dict["metrics/mAP50-95(B)"])
    }

def run_predictions_yolo(model, image_folder="downloads/yolo_dataset/images/val", output_folder="outputs/yolo", batch_size=1, num_images=40):
    os.makedirs(output_folder, exist_ok=True)
    image_paths = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(".png")])[:num_images]

    for img_path in image_paths:
        model.predict(img_path, save=True, save_txt=True, project=output_folder, name="predict", batch=batch_size)


def predict_yolo_onnx_tensor(tensor: torch.Tensor = torch.rand(2, 3, 640, 640),
                              model_path: str = "self_yolo11x.onnx"):
    input_np = tensor.detach().cpu().numpy()
    _, _, h, w = tensor.shape
    transform = LetterBox(new_shape=(h, w))
    processed = []
    for i in range(input_np.shape[0]):
        img_np = input_np[i].transpose(1, 2, 0)
        img_np = (img_np * 255).astype(np.uint8)
        resized = transform(image=img_np)
        resized = resized.astype(np.float32) / 255.0
        resized = resized.transpose(2, 0, 1)
        processed.append(resized)
    input_data = np.stack(processed, axis=0).astype(np.float32)
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_data})
    for i, out in enumerate(outputs):
        logging.info(f"Output[{i}] shape: {out.shape}")
    return outputs

class YoloCalibrationDataLoader(CalibrationDataReader):
    def __init__(self, image_dir, model_path, batch_size=8, num_batches=5, image_size=(640, 640)):
        self.image_paths = sorted(glob(os.path.join(image_dir, "*.*")))
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.image_size = image_size
        self.transform = LetterBox(new_shape=image_size)
        self.index = 0

        model = onnx.load(model_path)
        self.input_name = model.graph.input[0].name

    def __len__(self):
        return min(self.num_batches, (len(self.image_paths) + self.batch_size - 1) // self.batch_size)

    def reset(self):
        self.index = 0

    def get_next(self):
        if self.index >= len(self.image_paths) or self.index // self.batch_size >= self.num_batches:
            return None

        batch_paths = self.image_paths[self.index:self.index + self.batch_size]
        processed = []

        for path in batch_paths:
            img = cv2.imread(path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resized = self.transform(image=img)
            resized = resized.astype(np.float32) / 255.0
            resized = resized.transpose(2, 0, 1)  # HWC -> CHW
            processed.append(resized)

        self.index += self.batch_size

        if processed:
            batch_np = np.stack(processed, axis=0).astype(np.float32)
            return {self.input_name: batch_np}
        else:
            return None