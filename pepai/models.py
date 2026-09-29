"""Model registry: FP32 ONNX export and the matching input preprocessing."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.ops.misc import FrozenBatchNorm2d

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class ModelSpec:
    name: str
    task: str                 # "det" or "cls"
    input_name: str
    # (C, H, W) used for benchmarking; detection backbones accept dynamic H, W.
    bench_shape: tuple
    # Per-dimension (min, max) for the dynamic spatial axes, or None for fixed shapes.
    dynamic_hw: tuple | None = None


SPECS = {
    # Only the backbone + FPN of Faster R-CNN is exported and quantized;
    # RPN and ROI heads run in FP32 PyTorch (see frcnn_with_backbone).
    "frcnn_r50_fpn": ModelSpec("frcnn_r50_fpn", "det", "images", (3, 800, 1088), ((256, 1344), (256, 1344))),
    "yolov10s": ModelSpec("yolov10s", "det", "images", (3, 640, 640)),
    "yolov10x": ModelSpec("yolov10x", "det", "images", (3, 640, 640)),
    "efficientnet_b0": ModelSpec("efficientnet_b0", "cls", "images", (3, 224, 224)),
    "densenet121": ModelSpec("densenet121", "cls", "images", (3, 224, 224)),
}


def engine_shapes(spec, batch=1):
    """TensorRT optimization profile {input: (min, opt, max)} for batch-1 analysis engines."""
    c, h, w = spec.bench_shape
    if spec.dynamic_hw:
        (hmin, hmax), (wmin, wmax) = spec.dynamic_hw
        return {spec.input_name: ((batch, c, hmin, wmin), (batch, c, h, w), (batch, c, hmax, wmax))}
    return {spec.input_name: ((batch, c, h, w),) * 3}


# ---------------------------------------------------------------------------
# Faster R-CNN
# ---------------------------------------------------------------------------

def fold_frozen_bn(model):
    """Fold every FrozenBatchNorm2d of a torchvision ResNet into its preceding conv.

    Deployment runtimes (TensorRT, ONNX Runtime) always fold BN into the convolution,
    so doing it explicitly keeps the exported graph identical to what is executed.
    """
    def fold(conv, bn):
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        conv.weight.data = conv.weight.data * scale.reshape(-1, 1, 1, 1)
        bias = conv.bias.data if conv.bias is not None else torch.zeros_like(bn.bias)
        conv.bias = nn.Parameter(bn.bias + (bias - bn.running_mean) * scale)

    body = model.backbone.body
    fold(body.conv1, body.bn1)
    body.bn1 = nn.Identity()
    for layer in (body.layer1, body.layer2, body.layer3, body.layer4):
        for block in layer:
            for i in (1, 2, 3):
                fold(getattr(block, f"conv{i}"), getattr(block, f"bn{i}"))
                setattr(block, f"bn{i}", nn.Identity())
            if block.downsample is not None:
                fold(block.downsample[0], block.downsample[1])
                block.downsample[1] = nn.Identity()
    assert not any(isinstance(m, FrozenBatchNorm2d) for m in model.modules())
    return model


def load_frcnn():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    ).eval()
    return fold_frozen_bn(model)


class FRCNNBackbone(nn.Module):
    """ResNet-50 + FPN; outputs the five pyramid levels ('0'..'3', 'pool')."""

    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone

    def forward(self, images):
        return tuple(self.backbone(images).values())


FRCNN_OUTPUTS = ["feat0", "feat1", "feat2", "feat3", "feat_pool"]


def frcnn_preprocess(model, pil_images, device="cuda"):
    """Resize (min 800 / max 1333), normalise and pad to a multiple of 32, exactly as torchvision."""
    tensors = [torchvision.transforms.functional.to_tensor(im).to(device) for im in pil_images]
    image_list, _ = model.transform(tensors)
    return image_list


# ---------------------------------------------------------------------------
# YOLOv10 (Ultralytics, NMS-free end-to-end head)
# ---------------------------------------------------------------------------

def load_yolo(name, weights_dir):
    from ultralytics import YOLO
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    return YOLO(str(weights_dir / f"{name}.pt"))


def letterbox(pil_image, size=640, fill=114):
    """Ultralytics-style letterbox: keep aspect ratio, pad symmetrically. Returns array + (scale, pad)."""
    im = np.asarray(pil_image.convert("RGB"))
    h, w = im.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    resized = np.asarray(Image.fromarray(im).resize((nw, nh), Image.BILINEAR))
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas = np.full((size, size, 3), fill, dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = resized
    x = torch.from_numpy(canvas).permute(2, 0, 1).float() / 255.0
    return x, (r, left, top)


# ---------------------------------------------------------------------------
# ImageNet classifiers
# ---------------------------------------------------------------------------

def load_classifier(name):
    if name == "efficientnet_b0":
        w = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
        return torchvision.models.efficientnet_b0(weights=w).eval()
    if name == "densenet121":
        w = torchvision.models.DenseNet121_Weights.IMAGENET1K_V1
        return torchvision.models.densenet121(weights=w).eval()
    raise KeyError(name)


def classifier_preprocess(pil_image):
    t = torchvision.transforms
    tf = t.Compose([
        t.Resize(256, interpolation=t.InterpolationMode.BILINEAR),
        t.CenterCrop(224),
        t.ToTensor(),
        t.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return tf(pil_image.convert("RGB"))


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_onnx(name, out_path, weights_dir):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    spec = SPECS[name]

    if name == "frcnn_r50_fpn":
        net = FRCNNBackbone(load_frcnn()).eval()
        dummy = torch.randn(1, *spec.bench_shape)
        torch.onnx.export(
            net, (dummy,), str(out_path), input_names=["images"], output_names=FRCNN_OUTPUTS,
            dynamic_axes={"images": {0: "batch", 2: "height", 3: "width"},
                          **{o: {0: "batch", 2: f"h_{o}", 3: f"w_{o}"} for o in FRCNN_OUTPUTS}},
            opset_version=17, dynamo=False,
        )
    elif name.startswith("yolov10"):
        model = load_yolo(name, weights_dir)
        # nms=False selects the NMS-free one-to-one head of YOLOv10 (output: N x 300 x 6).
        exported = model.export(format="onnx", imgsz=640, dynamic=True, simplify=True, opset=17, nms=False)
        Path(exported).replace(out_path)
    else:
        net = load_classifier(name)
        dummy = torch.randn(1, *spec.bench_shape)
        torch.onnx.export(
            net, (dummy,), str(out_path), input_names=["images"], output_names=["logits"],
            dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17, dynamo=False,
        )
    return out_path
