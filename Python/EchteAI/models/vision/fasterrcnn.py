import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
import logging
import os
import torch
import numpy as np
import cv2
import torch.ao.quantization as quant
import EchteAI.models.quantized.quantized_resnet50 as qresnet

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

def compute_iou_fasterrcnn(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def compute_metrics_fasterrcnn(data_loader, model, device, iou_threshold=0.5):
    model.eval()
    total_gt = 0
    total_tp = 0
    total_pred = 0
    iou_list = []
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader):
            if batch_idx > 0:
                break
            images = [img.to(device) for img in images]
            predictions = model(images)
            for target, prediction in zip(targets, predictions):
                if "boxes" not in target:
                    continue
                gt_boxes = target["boxes"].cpu().numpy()
                gt_labels = target["labels"].cpu().numpy()
                total_gt += len(gt_boxes)
                pred_boxes = prediction["boxes"].cpu().numpy()
                pred_labels = prediction["labels"].cpu().numpy()
                pred_scores = prediction["scores"].cpu().numpy()
                keep = pred_scores >= 0.5
                pred_boxes = pred_boxes[keep]
                pred_labels = pred_labels[keep]
                total_pred += len(pred_boxes)
                matched = [False] * len(gt_boxes)
                for pb, pl in zip(pred_boxes, pred_labels):
                    best_iou = 0
                    best_idx = -1
                    for i, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
                        if matched[i]:
                            continue
                        if pl != gl:
                            continue
                        iou = compute_iou_fasterrcnn(pb, gb)
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = i
                    if best_iou >= iou_threshold and best_idx != -1:
                        matched[best_idx] = True
                        total_tp += 1
                        iou_list.append(best_iou)
    accuracy = total_tp / total_gt if total_gt > 0 else 0
    precision = total_tp / total_pred if total_pred > 0 else 0
    mean_iou = sum(iou_list) / len(iou_list) if iou_list else 0
    return {"accuracy": accuracy, "precision": precision, "mean_iou": mean_iou}

def compute_batch_metrics_fasterrcnn(targets, predictions, iou_threshold=0.5):
    total_gt = 0
    total_tp = 0
    total_pred = 0
    iou_list = []
    for target, prediction in zip(targets, predictions):
        if "boxes" not in target:
            continue
        gt_boxes = target["boxes"].cpu().numpy()
        gt_labels = target["labels"].cpu().numpy()
        total_gt += len(gt_boxes)
        pred_boxes = prediction["boxes"].cpu().numpy()
        pred_labels = prediction["labels"].cpu().numpy()
        pred_scores = prediction["scores"].cpu().numpy()
        keep = pred_scores >= 0.5
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        total_pred += len(pred_boxes)
        matched = [False] * len(gt_boxes)
        for pb, pl in zip(pred_boxes, pred_labels):
            best_iou = 0
            best_idx = -1
            for i, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
                if matched[i]:
                    continue
                if pl != gl:
                    continue
                iou = compute_iou_fasterrcnn(pb, gb)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_iou >= iou_threshold and best_idx != -1:
                matched[best_idx] = True
                total_tp += 1
                iou_list.append(best_iou)
    accuracy = total_tp / total_gt if total_gt > 0 else 0
    precision = total_tp / total_pred if total_pred > 0 else 0
    mean_iou = sum(iou_list) / len(iou_list) if iou_list else 0
    return {"accuracy": accuracy, "precision": precision, "mean_iou": mean_iou}

def train_fasterrcnn(model, train_loader, val_loader, device, num_epochs, model_path="model.pth"):
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        logging.info(f"Loaded saved model from {model_path}.")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00001)
        logging.info("Training started.")
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0
            for batch_idx, (images, targets) in enumerate(train_loader):
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                running_loss += losses.item()
                with torch.no_grad():
                    model.eval()
                    predictions = model(images)
                    batch_metrics = compute_batch_metrics_fasterrcnn(targets, predictions)
                    model.train()
                logging.debug(
                    f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {losses.item():.4f}, "
                    f"Acc: {batch_metrics['accuracy']:.4f}, Prec: {batch_metrics['precision']:.4f}, "
                    f"mIoU: {batch_metrics['mean_iou']:.4f}"
                )
            avg_loss = running_loss / len(train_loader)
            train_metrics = compute_metrics_fasterrcnn(train_loader, model, device)
            val_metrics = compute_metrics_fasterrcnn(val_loader, model, device)
            logging.info(
                f"Epoch {epoch+1}/{num_epochs} finished, avg loss: {avg_loss:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f}, Train Prec: {train_metrics['precision']:.4f}, "
                f"Train mIoU: {train_metrics['mean_iou']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, "
                f"Val Prec: {val_metrics['precision']:.4f}, Val mIoU: {val_metrics['mean_iou']:.4f}"
            )
        logging.info("Training finished.")
        torch.save(model.state_dict(), model_path)
    return model

def run_predictions_fasterrcnn(model, data_loader, device, dataset, output_folder, evaluate=False, num_batches = -1):
    os.makedirs(output_folder, exist_ok=True)
    model.to(device)
    model.eval()
    if evaluate:
        metrics = compute_metrics_fasterrcnn(data_loader, model, device)
        logging.info(f"Metrics on dataset: Acc: {metrics['accuracy']:.4f}, Prec: {metrics['precision']:.4f}, mIoU: {metrics['mean_iou']:.4f}")
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader):
            if num_batches != 0 and batch_idx > num_batches - 1:
                break
            images = [img.to(device) for img in images]
            predictions = model(images)
            for i, (img, prediction) in enumerate(zip(images, predictions)):
                image_np = img.mul(255).byte().permute(1, 2, 0).cpu().numpy()
                image_np = np.ascontiguousarray(image_np)
                if image_np.dtype != np.uint8:
                    image_np = image_np.astype(np.uint8)
                
                # Ground Truth -> Red
                if "boxes" in targets[i]:
                    for box in targets[i]["boxes"]:
                        x1, y1, x2, y2 = map(int, box.tolist())
                        cv2.rectangle(image_np, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # Predictions -> Green
                for j, box in enumerate(prediction["boxes"]):
                    score = prediction["scores"][j].item()
                    if score < 0.5:
                        continue
                    x1, y1, x2, y2 = map(int, box.tolist())
                    label_int = prediction["labels"][j].item()
                    label_name = dataset.idx_to_class.get(label_int, "Unknown")
                    cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image_np, f"{label_name}: {score:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                output_path = os.path.join(output_folder, f"batch{batch_idx}_img{i}.png")
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, image_bgr)
            logging.info(f"Batch {batch_idx} saved.")

def quantize_dynamic(model):
    model.to("cpu")
    model.eval()

    model_quantized = torch.ao.quantization.quantize_dynamic(model)
    logging.info("Dynamic quantization applied (qint8_dynamic).")
    
    model_quantized.eval()
    return model_quantized.to("cpu")

def quantize_fasterrcnn(model_fp32, data_loader, number_of_batches=2):
    model_fp32.eval().to("cpu")
    logging.info("Loading quantized backbone...")

    quantized_model = qresnet.quantized_resnet50()
    state_dict = model_fp32.backbone.body.state_dict()
    quantized_model.load_state_dict(state_dict, strict=False)
    quantized_model.eval()

    activation_observer = quant.MinMaxObserver.with_args(dtype=torch.quint8, quant_min=0, quant_max=255)
    weight_observer = quant.default_per_channel_weight_observer
    quantized_model.qconfig = quant.QConfig(
        activation=activation_observer,
        weight=weight_observer
    )

    logging.info("Calibrate quantized backbone...")

    quant.prepare(quantized_model, inplace=True)

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(data_loader):
            if batch_idx >= number_of_batches:
                break

            transformed_batch, _ = model_fp32.transform(images)

            if isinstance(transformed_batch, torchvision.models.detection.image_list.ImageList):
                images_tensor = transformed_batch.tensors
            else:
                raise TypeError(f"Nem megfelelő transzformált batch formátum: {type(transformed_batch)}")

            quantized_model(images_tensor)

    quant.convert(quantized_model, inplace=True)
    quantized_model.eval()

    model_fp32.backbone.body = quantized_model
    model_fp32.eval()

    logging.info("Loaded quantized backbone.")

    return model_fp32
