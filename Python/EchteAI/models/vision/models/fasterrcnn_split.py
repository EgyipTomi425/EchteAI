import os

import torch
import logging
import time
from collections import OrderedDict
from torchvision.models.detection.image_list import ImageList
import torch.nn as nn
import onnxruntime as ort
import numpy as np

class FeatureExtractor(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, images):
        features = self.backbone(images)
        feats = list(features.values())
        return (
            feats[0], feats[1], feats[2], feats[3], feats[4],
        )

class DetectorHead(nn.Module):
    def __init__(self, rpn, roi_heads, postprocess):
        super().__init__()
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.postprocess = postprocess

    def forward(self, tensors, image_sizes, original_sizes, feat0, feat1, feat2, feat3, feat4):
        feats = {"0": feat0, "1": feat1, "2": feat2, "3": feat3, "4": feat4}
        img_list = ImageList(tensors, image_sizes)
        proposals, _ = self.rpn(img_list, feats, targets=None)
        detections, _ = self.roi_heads(feats, proposals, image_sizes, targets=None)
        results = self.postprocess(detections, image_sizes, original_sizes)
        boxes = [r["boxes"] for r in results]
        labels = [r["labels"] for r in results]
        scores = [r["scores"] for r in results]
        return boxes, labels, scores

def split_frcnn_pipeline(model, images, device):
    logging.info("=== Comparison: full forward vs. manual pipeline (batch) ===")
    model.eval()
    images = [img.to(device) for img in images]
    fe = FeatureExtractor(model.backbone).to(device)
    dh = DetectorHead(model.rpn, model.roi_heads, model.transform.postprocess).to(device)
    with torch.no_grad():
        t0 = time.time()
        full_output = model(images)
        t1 = time.time()

        t2 = time.time()
        original_sizes = torch.tensor([list(img.shape[-2:]) for img in images], dtype=torch.int64)
        img_list, _ = model.transform(images)
        tensors = img_list.tensors
        image_sizes = torch.tensor([list(s) for s in img_list.image_sizes], dtype=torch.int64)

        f0, f1, f2, f3, f4 = fe(tensors)
        manual_boxes, manual_labels, manual_scores = dh(tensors, image_sizes, original_sizes, f0, f1, f2, f3, f4)
        t3 = time.time()

    logging.info(f"⏱️ Full forward time:     {(t1 - t0):.4f} s")
    logging.info(f"⏱️ Manual pipeline time:  {(t3 - t2):.4f} s")
    for i, full in enumerate(full_output):
        logging.info(f"\n📷 Image {i}:")
        logging.info(f"📦 Box count - Full: {len(full['boxes'])}, Manual: {len(manual_boxes[i])}")
        if len(full['boxes']) == len(manual_boxes[i]) and len(full['boxes']) > 0:
            box_diff = torch.abs(full['boxes'] - manual_boxes[i]).mean().item()
            score_diff = torch.abs(full['scores'] - manual_scores[i]).mean().item()
            label_diff = int((full['labels'] != manual_labels[i]).sum().item())
            logging.info(f"🔢 Avg. box difference:   {box_diff:.6f}")
            logging.info(f"🔢 Avg. score difference: {score_diff:.6f}")
            logging.info(f"❗ Label mismatches:      {label_diff}")
        else:
            logging.warning("⚠️ Box count mismatch or empty detections.")
            logging.warning(f"  Full boxes: {len(full['boxes'])}, Manual boxes: {len(manual_boxes[i])}")
    return fe, dh

def split_save_frcnn(model, images, device, model_dir="./outputs/models"):
    fe_path = os.path.join(model_dir, "feature_extractor.onnx")
    dh_path = os.path.join(model_dir, "detector_head.onnx")
    logging.info("🔧 Preparing model and inputs")
    fe, dh = split_frcnn_pipeline(model, images, device)
    fe.eval(); dh.eval()

    original_sizes = torch.tensor([list(img.shape[-2:]) for img in images], dtype=torch.int64)
    img_list, _ = model.transform(images)
    tensors = img_list.tensors
    image_sizes = torch.tensor([list(s) for s in img_list.image_sizes], dtype=torch.int64)
    
    dynamic_shapes = {"images": {0: "batch", 2: "h", 3: "w"}}
    torch.onnx.export(
        fe,
        (tensors,),
        fe_path,
        input_names=["images"],
        output_names=["feat0", "feat1", "feat2", "feat3", "feat4"],
        dynamic_axes=dynamic_shapes,
        verbose=False,
        dynamo=True
    )

    with torch.no_grad():
        f0, f1, f2, f3, f4 = fe(tensors)

    torch.onnx.export(
        dh,
        (tensors, image_sizes, original_sizes, f0, f1, f2, f3, f4),
        dh_path,
        input_names=["tensors", "image_sizes", "orig_sizes", "feat0", "feat1", "feat2", "feat3", "feat4"],
        output_names=["boxes", "labels", "scores"],
        dynamic_axes={
            "tensors": {0: "batch", 2: "height", 3: "width"},
            "feat0": {0: "batch", 2: "h", 3: "w"},
            "feat1": {0: "batch", 2: "h", 3: "w"},
            "feat2": {0: "batch", 2: "h", 3: "w"},
            "feat3": {0: "batch", 2: "h", 3: "w"},
            "feat4": {0: "batch", 2: "h", 3: "w"},
        },
        verbose=False,
        #dynamo=True # cannot symbolic trace if RPN/ROI heads use control flow, so we skip dynamo for now
    )

class ONNXFasterRCNNWrapper(torch.nn.Module):
    def __init__(self, fe_onnx_path="feature_extractor.onnx", dh_onnx_path="detector_head.onnx", transform=None, device=None):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(device, torch.device):
            device = device.type
        else:
            device = str(device).lower()
        self.device = device
        self.transform = transform
        sess_options = ort.SessionOptions()
        if "cpu" in device:
            sess_options.enable_mem_pattern = False
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess_options.enable_cpu_mem_arena = False
        providers = ['CUDAExecutionProvider'] if "cuda" in device else ['CPUExecutionProvider']
        self.fe_session = ort.InferenceSession(fe_onnx_path, providers=providers, sess_options=sess_options)
        self.dh_session = ort.InferenceSession(dh_onnx_path, providers=providers, sess_options=sess_options)

    def forward(self, images):
        original_sizes = torch.tensor([list(img.shape[-2:]) for img in images], dtype=torch.int64)
        img_list, _ = self.transform(images)
        tensors = img_list.tensors
        image_sizes = torch.tensor([list(s) for s in img_list.image_sizes], dtype=torch.int64)
        fe_outputs = self.fe_session.run(None, {"images": tensors.cpu().numpy()})
        fe_names = [o.name for o in self.fe_session.get_outputs()]
        fe_dict = dict(zip(fe_names, fe_outputs))

        dh_inputs = {
            "tensors": tensors.cpu().numpy(),
            "image_sizes": image_sizes.cpu().numpy(),
            "orig_sizes": original_sizes.cpu().numpy(),
            "feat0": fe_dict["feat0"],
            "feat1": fe_dict["feat1"],
            "feat2": fe_dict["feat2"],
            "feat3": fe_dict["feat3"],
            "feat4": fe_dict["feat4"],
        }

        dh_outputs = self.dh_session.run(None, dh_inputs)

        bs = len(images)
        boxes  = dh_outputs[0:bs]
        labels = dh_outputs[bs:2*bs]
        scores = dh_outputs[2*bs:3*bs]

        results = []
        for i in range(bs):
            result = {
                "boxes": torch.tensor(boxes[i], device=self.device),
                "labels": torch.tensor(labels[i], device=self.device),
                "scores": torch.tensor(scores[i], device=self.device),
            }
            results.append(result)
        return results