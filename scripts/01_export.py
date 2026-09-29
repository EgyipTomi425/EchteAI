"""Export every model to FP32 ONNX and check the Faster R-CNN BN folding numerically."""
import torch
import torchvision
from PIL import Image

from pepai.config import load_config, results_dir
from pepai.models import export_onnx, frcnn_preprocess, load_frcnn


def check_bn_folding(cfg):
    ref = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="COCO_V1").eval().cuda()
    folded = load_frcnn().cuda()
    img = Image.open(sorted((cfg["coco"]["root"] / "val2017").glob("*.jpg"))[0]).convert("RGB")
    x = frcnn_preprocess(ref, [img]).tensors
    with torch.no_grad():
        a = list(ref.backbone(x).values())
        b = list(folded.backbone(x).values())
    worst = max(((p - q).abs().max() / p.abs().max()).item() for p, q in zip(a, b))
    print(f"BN folding: max relative deviation of FPN outputs = {worst:.2e}")
    assert worst < 1e-4


if __name__ == "__main__":
    cfg = load_config()
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    check_bn_folding(cfg)
    onnx_dir = results_dir(cfg, "onnx")
    for name in cfg["models"]:
        path = onnx_dir / f"{name}_fp32.onnx"
        if not path.exists():
            export_onnx(name, path, weights_dir=cfg["paths"]["data"] / "weights")
        print(f"{name}: {path} ({path.stat().st_size / 1e6:.1f} MB)")
