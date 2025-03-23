import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
import logging

def setup_fasterrcnn(dataset=None, backbone="resnet50"):
    model_choices = {
        "resnet50": (fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights.DEFAULT),
        "mobilenet_v3": (fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
    }

    if backbone not in model_choices:
        raise ValueError(f"Unknown backbone: {backbone}. Use a valid one: {list(model_choices.keys())}")

    logging.info(f"Loading {backbone}...")

    model_fn, weights = model_choices[backbone]
    model = model_fn(weights=weights)

    if dataset is not None:
        num_classes = len(dataset.dataset.class_to_idx) + 1 if hasattr(dataset, "dataset") else len(dataset.class_to_idx) + 1
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    logging.info(f"The {backbone} (faster-r-cnn model) is ready.")

    return model
