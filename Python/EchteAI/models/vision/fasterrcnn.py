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
import time
import matplotlib.pyplot as plt
from torchvision.models.detection.image_list import ImageList

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
            start_time = time.time()
            predictions = model(images)
            batch_time = time.time() - start_time
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
            logging.info(f"Batch {batch_idx} processed in {batch_time:.4f} seconds.")

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

    quantized_resnet = qresnet.quantized_resnet50()
    resnet_state_dict = model_fp32.backbone.body.state_dict()
    quantized_resnet.load_state_dict(resnet_state_dict, strict=False)
    quantized_resnet.eval()

    activation_observer = quant.HistogramObserver.with_args(dtype=torch.quint8, quant_min=0, quant_max=255)
    weight_observer = quant.default_per_channel_weight_observer
    quantized_resnet.qconfig = quant.QConfig(
        activation=activation_observer,
        weight=weight_observer
    )

    logging.info("Calibrate quantized backbone...")

    quant.prepare(quantized_resnet, inplace=True)

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(data_loader):
            if batch_idx >= number_of_batches and number_of_batches >= 0:
                break

            transformed_batch, _ = model_fp32.transform(images)

            if isinstance(transformed_batch, torchvision.models.detection.image_list.ImageList):
                images_tensor = transformed_batch.tensors
            else:
                raise TypeError(f"Nem megfelelő transzformált batch formátum: {type(transformed_batch)}")

            quantized_resnet(images_tensor)

    quant.convert(quantized_resnet, inplace=True)
    quantized_resnet.eval()

    import copy
    model_quantized = copy.deepcopy(model_fp32)
    model_quantized.backbone.body = quantized_resnet
    model_quantized.eval()

    logging.info("Loaded quantized backbone.")

    return model_quantized

def backbone_cnn_layers_outputs(model_quantized, image=torch.randn(1000, 500)):
    hooks = []
    outputs = {}
    
    for name, layer in model_quantized.backbone.body.named_modules():
        if isinstance(layer, (torch.ao.nn.quantized.Conv2d)):
            hook = layer.register_forward_hook(
                lambda m, i, o, name=name: outputs.update({name: torch.dequantize(o)})
            )
            hooks.append(hook)
        elif isinstance(layer, (torch.nn.Conv2d)):
            hook = layer.register_forward_hook(
                lambda m, i, o, name=name: outputs.update({name: o})
            )
            hooks.append(hook)
    
    output = model_quantized(image.unsqueeze(0))
    
    for hook in hooks:
        hook.remove()

    return outputs

def absolute_differences(outputs1, outputs2):
    abs_diffs = {}
    for key in outputs1:
        if key in outputs2:
            if outputs1[key].shape == outputs2[key].shape:
                diff = torch.abs(outputs1[key] - outputs2[key])
                abs_diffs[key] = diff
            else:
                print(f"Shape mismatch at layer '{key}', skipping.")
        else:
            print(f"Layer '{key}' not found in both outputs.")
    return abs_diffs

def percentage_differences(outputs1, outputs2):
    percent_diffs = {}
    for key in outputs1:
        if key in outputs2:
            if outputs1[key].shape == outputs2[key].shape:
                diff = torch.abs(outputs1[key] - outputs2[key])
                base = torch.abs(outputs1[key])
                percent = torch.where(base == 0, torch.ones_like(base), diff / base)
                percent_diffs[key] = percent
            else:
                print(f"Shape mismatch at layer '{key}', skipping.")
        else:
            print(f"Layer '{key}' not found in both outputs.")
    return percent_diffs

def visualize_cnn_outputs(outputs, output_folder="outputs", filename="activation_heatmap", vmin=None, vmax=None, depth=-1, layer=None):
    os.makedirs(output_folder, exist_ok=True)

    output_items = list(outputs.items())

    largest_shape = max([feat.shape[-2:] for _, feat in output_items])
    logging.debug(f"Largest shape is: {largest_shape}.")

    heatmap = np.zeros(largest_shape, dtype=np.float32)
    weight_sum = np.zeros(largest_shape, dtype=np.float32)

    if layer is not None:
        if not (0 < layer < len(output_items)):
            raise ValueError(f"Invalid layer index: {layer}. It must be between 1 and {len(output_items)}.")
        output_items = [output_items[layer-1]]

    for name, feature_map in output_items:
        feature_map = feature_map.cpu().detach().numpy()

        if feature_map.ndim == 4:
            avg_map = np.max(feature_map, axis=(0, 1)).squeeze()
        elif feature_map.ndim == 3:
            avg_map = np.max(feature_map, axis=0).squeeze()
        else:
            raise ValueError(f"Unexpected feature map shape for layer '{name}': {feature_map.shape}")

        resized_map = cv2.resize(avg_map, (largest_shape[1], largest_shape[0]), interpolation=cv2.INTER_CUBIC)
        heatmap += resized_map
        weight_sum += np.ones_like(resized_map)

        if depth != -1 and depth == 0:
            break
        else:
            depth -= 1

    heatmap /= np.maximum(weight_sum, 1e-6)

    if vmin is None or vmax is None:
        vmin, vmax = heatmap.min(), heatmap.max()

    heatmap = np.clip((heatmap - vmin) / (vmax - vmin), 0, 1)
    heatmap = np.uint8(heatmap * 255)
    logging.debug(f"The Heatmap: {heatmap}")
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    output_path = os.path.join(output_folder, f"{filename}.png")
    cv2.imwrite(output_path, heatmap)
    logging.info(f"Picture saved: {output_path}.")

def fit_and_plot_distribution(outputs1, outputs2, output_folder="outputs", filename="distribution_fit", layer=-1, depth=-1):
    os.makedirs(output_folder, exist_ok=True)

    keys = list(outputs1.keys())
    if layer != -1:
        selected_keys = [keys[layer - 1]]
    elif depth != -1:
        selected_keys = keys[:depth]
    else:
        selected_keys = keys

    x_vals = []
    y_vals = []

    for key in selected_keys:
        if key in outputs2 and outputs1[key].shape == outputs2[key].shape:
            base = outputs1[key].flatten().cpu().numpy()
            diff = outputs2[key].flatten().cpu().numpy()

            x_vals.append(base)
            y_vals.append(diff)

    if not x_vals or not y_vals:
        print("No valid layers found to plot.")
        return

    x_vals = np.concatenate(x_vals)
    y_vals = np.concatenate(y_vals)

    sort_idx = np.argsort(x_vals)
    x_sorted = x_vals[sort_idx]
    y_sorted = y_vals[sort_idx]

    plt.figure(figsize=(12, 6))
    plt.scatter(x_sorted, y_sorted, s=2, alpha=0.3, label="Data", color="gray")

    poly_coeffs = np.polyfit(x_sorted, y_sorted, 10)
    poly = np.poly1d(poly_coeffs)

    plt.plot(x_sorted, poly(x_sorted), label="10th degree polynomial", color="purple", linestyle="--")

    plt.xlabel("Original activations")
    plt.ylabel("Difference (abs or percent)")
    plt.legend()
    plt.title("Polynomial Fit to All Data Points")
    plt.tight_layout()

    path_1 = os.path.join(output_folder, f"{filename}_polynomial.png")
    plt.savefig(path_1)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.scatter(x_vals, y_vals, s=2, alpha=0.3, color="gray")
    plt.xlabel("Original activations")
    plt.ylabel("Difference (abs or percent)")
    plt.title("Scatter Plot of All Points")
    plt.tight_layout()

    path_2 = os.path.join(output_folder, f"{filename}_scatter.png")
    plt.savefig(path_2)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.hexbin(x_vals, y_vals, gridsize=50, cmap='Blues', bins='log')
    plt.colorbar(label='Log Density')
    plt.xlabel("Original activations")
    plt.ylabel("Difference (abs or percent)")
    plt.title("2D Density Distribution (Hexbin)")
    plt.tight_layout()

    path_3 = os.path.join(output_folder, f"{filename}_distribution.png")
    plt.savefig(path_3)
    plt.close()

def compare_models_visual(model1, model2, data_loader, device, dataset, output_folder, num_batches=1):

    os.makedirs(output_folder, exist_ok=True)
    model1.eval().to(device)
    model2.eval().to(device)

    static_vmin, static_vmax = None, None

    def get_feature_heatmap(feats, img_shape):
        heatmap = None
        count = 0
        for name, fmap in feats.items():
            if isinstance(fmap, torch.Tensor):
                fmap = fmap.detach().cpu()
            if fmap.ndim == 4:
                fmap = fmap.squeeze(0)
            if fmap.ndim == 3:
                fmap = torch.max(fmap, dim=0).values
            elif fmap.ndim != 2:
                continue

            fmap_np = fmap.numpy()
            if np.any(np.isnan(fmap_np)) or np.any(np.isinf(fmap_np)):
                continue

            resized = cv2.resize(fmap_np, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_CUBIC)
            heatmap = resized if heatmap is None else heatmap + resized
            count += 1

        if heatmap is not None and count > 0:
            heatmap /= count
        else:
            heatmap = np.zeros(img_shape, dtype=np.float32)

        return heatmap

    def extract_conv_features(model, image_tensor):
        features = {}
        hooks = []

        def register_hooks(module, name):
            if isinstance(module, (torch.nn.Conv2d, torch.ao.nn.quantized.Conv2d)):
                hooks.append(
                    module.register_forward_hook(
                        lambda m, i, o: features.update({name: torch.dequantize(o) if hasattr(o, "dequantize") else o})
                    )
                )

        for name, module in model.backbone.body.named_modules():
            register_hooks(module, name)

        with torch.no_grad():
            model(image_tensor.unsqueeze(0).to("cpu"))

        for hook in hooks:
            hook.remove()
        return features

    def normalize_heatmap(hmap, vmin, vmax):
        hmap = np.clip((hmap - vmin) / (vmax - vmin + 1e-5), 0, 1)
        hmap = np.uint8(hmap * 255)
        return cv2.applyColorMap(hmap, cv2.COLORMAP_JET)

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader):
            if batch_idx >= num_batches && num_batches > -1:
                break
            images = [img.to(device) for img in images]
            predictions1 = model1(images)
            predictions2 = model2(images)

            for i, (img_tensor, pred1, pred2) in enumerate(zip(images, predictions1, predictions2)):
                image_np = img_tensor.mul(255).byte().permute(1, 2, 0).cpu().numpy()
                image_np = np.ascontiguousarray(image_np)
                h, w, _ = image_np.shape

                vis_pred1 = image_np.copy()
                vis_pred2 = image_np.copy()

                if "boxes" in targets[i]:
                    for box in targets[i]["boxes"]:
                        x1, y1, x2, y2 = map(int, box.tolist())
                        cv2.rectangle(vis_pred1, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.rectangle(vis_pred2, (x1, y1), (x2, y2), (255, 0, 0), 2)

                for box, score, label in zip(pred1["boxes"], pred1["scores"], pred1["labels"]):
                    if score < 0.5:
                        continue
                    x1, y1, x2, y2 = map(int, box.tolist())
                    label_name = dataset.idx_to_class.get(label.item(), "Unknown")
                    cv2.rectangle(vis_pred1, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis_pred1, f"{label_name}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                for box, score, label in zip(pred2["boxes"], pred2["scores"], pred2["labels"]):
                    if score < 0.5:
                        continue
                    x1, y1, x2, y2 = map(int, box.tolist())
                    label_name = dataset.idx_to_class.get(label.item(), "Unknown")
                    cv2.rectangle(vis_pred2, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis_pred2, f"{label_name}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                feats1 = extract_conv_features(model1, img_tensor.cpu())
                feats2 = extract_conv_features(model2, img_tensor.cpu())

                feat1_map = get_feature_heatmap(feats1, (h, w))
                feat2_map = get_feature_heatmap(feats2, (h, w))

                if static_vmin is None or static_vmax is None:
                    static_vmin = feat1_map.min()
                    static_vmax = feat1_map.max()

                feat1_colormap = normalize_heatmap(feat1_map, static_vmin, static_vmax)
                feat2_colormap = normalize_heatmap(feat2_map, static_vmin, static_vmax)

                pred1_bgr = cv2.cvtColor(vis_pred1, cv2.COLOR_RGB2BGR)
                pred2_bgr = cv2.cvtColor(vis_pred2, cv2.COLOR_RGB2BGR)

                top_row = np.hstack((pred1_bgr, pred2_bgr))
                bottom_row = np.hstack((feat1_colormap, feat2_colormap))
                combined = np.vstack((top_row, bottom_row))

                output_path = os.path.join(output_folder, f"batch{batch_idx}_img{i}.png")
                cv2.imwrite(output_path, combined)
