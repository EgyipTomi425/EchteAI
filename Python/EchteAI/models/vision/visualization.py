import os
import cv2
import numpy as np
import torch
import logging
import matplotlib.pyplot as plt

def visualize_cnn_outputs(outputs, output_folder="outputs", filename="activation_heatmap", vmin=None, vmax=None, depth=-1, layer=None):
    os.makedirs(output_folder, exist_ok=True)

    output_items = list(outputs.items())

    largest_shape = max([feat.shape[-2:] for _, feat in output_items])
    logging.info(f"Largest shape is: {largest_shape}.")

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

def visualize_cnn_batch_outputs(outputs, output_folder="outputs", filename="activation_heatmap",
                                vmin=None, vmax=None, depth=-1, layer=None):
    os.makedirs(output_folder, exist_ok=True)

    outputs = {k: v for k, v in outputs.items() if torch.tensor(v).ndim == 4}
    output_items = list(outputs.items())

    if not output_items:
        raise ValueError("Nincs 4D-s (batch-es) feature map az outputok között.")

    batch_size = next(iter(outputs.values())).shape[0]

    for batch_idx in range(batch_size):
        single_outputs = {
            name: torch.tensor(fmap[batch_idx:batch_idx+1]) for name, fmap in outputs.items()
        }

        current_filename = f"{filename}_b{batch_idx}"
        visualize_cnn_outputs(
            outputs=single_outputs,
            output_folder=output_folder,
            filename=current_filename,
            vmin=vmin,
            vmax=vmax,
            depth=depth,
            layer=layer
        )

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
            if batch_idx >= num_batches and num_batches > -1:
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

def visualize_onnx_cnn_outputs(model_path, input_tensor, output_folder="outputs", filename_prefix="activation", vmin=None, vmax=None, depth=-1, layer=None):
    os.makedirs(output_folder, exist_ok=True)

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor.cpu().numpy()})
    output_names = [o.name for o in session.get_outputs()]
    output_items = list(zip(output_names, outputs))

    # (B, C, H, W)
    output_items = [(k, v) for k, v in output_items if torch.tensor(v).ndim == 4]
    if not output_items:
        raise ValueError("Nincs megfelelő 4D-s konvolúciós output a modellben.")

    if layer is not None:
        if not (0 < layer <= len(output_items)):
            raise ValueError(f"Invalid layer index: {layer}. It must be between 1 and {len(output_items)}.")
        output_items = [output_items[layer - 1]]

    batch_size = input_tensor.shape[0]
    largest_shape = max([v.shape[-2:] for _, v in output_items])
    logging.info(f"Largest shape is: {largest_shape}.")

    for i in range(batch_size):
        heatmap = np.zeros(largest_shape, dtype=np.float32)
        weight_sum = np.zeros(largest_shape, dtype=np.float32)
        depth_counter = depth

        for name, feature_map in output_items:
            fmap = torch.tensor(feature_map[i])  # [C, H, W]
            if fmap.ndim != 3:
                continue
            fmap = fmap.cpu().detach().numpy()
            avg_map = np.max(fmap, axis=0)

            resized_map = cv2.resize(avg_map, (largest_shape[1], largest_shape[0]), interpolation=cv2.INTER_CUBIC)
            heatmap += resized_map
            weight_sum += 1

            if depth_counter == 1:
                break
            elif depth_counter > 0:
                depth_counter -= 1

        heatmap /= np.maximum(weight_sum, 1e-6)

        if vmin is None or vmax is None:
            vmin_, vmax_ = heatmap.min(), heatmap.max()
        else:
            vmin_, vmax_ = vmin, vmax

        heatmap = np.clip((heatmap - vmin_) / (vmax_ - vmin_), 0, 1)
        heatmap = np.uint8(heatmap * 255)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        output_path = os.path.join(output_folder, f"{filename_prefix}_{i}.png")
        cv2.imwrite(output_path, heatmap)
        logging.info(f"Saved heatmap for image {i}: {output_path}")

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