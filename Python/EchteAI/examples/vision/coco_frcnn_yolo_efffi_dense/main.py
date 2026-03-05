# https://cocodataset.org/#download
# Dataset 2017

import warnings

from EchteAI.models.vision.models.onnx_quant import SingleImageCalibrationReader, quantize_onnx_model_calibdl_int8
import torchvision.utils as vutils
import onnxruntime as ort
import torch.nn.functional as F
import numpy as np

from EchteAI.models.vision.models.onnx_frcnn import onnx_conv_outputs_from_batch, quantize_feature_extractor
from EchteAI.models.vision.visualization import absolute_differences, compare_models_visual, fit_and_plot_distribution, visualize_cnn_outputs
warnings.filterwarnings("ignore")

import logging
import os
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split

import EchteAI.data.dataloaders as dl
from EchteAI.models.vision.models.fasterrcnn_split import ONNXFasterRCNNWrapper, split_save_frcnn
import EchteAI.models.vision.models.fasterrcnn_utils as frcnn_utils

from effdet import create_model

torch.manual_seed(42)

device = "cuda"#torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    cwd = os.path.dirname(__file__)

    image_dir = os.path.join(cwd, "downloads", "val2017")
    annotations_path = os.path.join(
        cwd,
        "downloads",
        "annotations",
        "instances_val2017.json"
    )

    output_dir_fp32 = os.path.join(
        cwd,
        "outputs",
        "frcnn",
        "val_fp32"
    )

    model_dir = os.path.join(cwd, "outputs", "models")
    os.makedirs(model_dir, exist_ok=True)

    if not os.path.exists(image_dir):
        logging.error(f"Input image folder not found: {image_dir}")
        return

    batch_size = 1

    dataset = dl.CocoDetectionDataset(
        image_dir=image_dir,
        annotation_path=annotations_path,
        transforms=T.Compose([T.ToTensor()])
    )

    total_len = len(dataset)
    calib_len = int(0.512 * total_len)
    val_len = total_len - calib_len

    calib_dataset, val_dataset = random_split(dataset, [calib_len, val_len])

    calib_loader = DataLoader(
        calib_dataset, batch_size=4, shuffle=True, collate_fn=lambda batch: tuple(zip(*batch))
    )
    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False, collate_fn=lambda batch: tuple(zip(*batch))
    )

    print(f"Calibration size: {len(calib_dataset)}, Validation size: {len(val_dataset)}")

    logging.info("Loading FP32 model...")
    model_fp32 = frcnn_utils.setup_fasterrcnn(backbone="resnet50")

    model_fp32.to(device).eval()

    logging.info("Running predictions on original FP32 model...")

    if False:
        frcnn_utils.run_predictions_fasterrcnn(
            model=model_fp32,
            data_loader=val_loader,
            device=device,
            dataset=dataset,
            output_folder=output_dir_fp32,
            evaluate=False,
            num_batches=16,
            score_threshold=0.80
        )

    logging.info("Exporting split FasterRCNN to ONNX...")

    images, _ = next(iter(calib_loader))
    images, _ = next(iter(calib_loader))
    images, _ = next(iter(calib_loader))
    images, _ = next(iter(calib_loader))
    calib_images = [img.to(device) for img in images]

    if False:
        split_save_frcnn(
            model=model_fp32,
            images=calib_images,
            device=device,
            model_dir=model_dir
        )

    logging.info("ONNX export finished.")
    fe_onnx_path = os.path.join(model_dir, "feature_extractor.onnx")
    dh_onnx_path = os.path.join(model_dir, "detector_head.onnx")

    onnx_model = ONNXFasterRCNNWrapper(
        fe_onnx_path=fe_onnx_path,
        dh_onnx_path=dh_onnx_path,
        transform=model_fp32.transform,
        device=device
    )

    output_dir_onnx_fp32 = os.path.join(
        cwd,
        "outputs",
        "frcnn",
        "onnx_val_fp32"
    )

    if False:
        frcnn_utils.run_predictions_fasterrcnn(
            model=onnx_model,
            data_loader=val_loader,
            device=device,
            dataset=dataset,
            output_folder=output_dir_onnx_fp32,
            evaluate=False,
            num_batches=16,
            score_threshold=0.80
        )
    
    if False:
        quantize_feature_extractor(fe_onnx_path, calib_loader, model_fp32.transform, os.path.join(model_dir, "feature_extractor_quant.onnx"), num_batches=8)

    quantized_fe_path = os.path.join(model_dir, "feature_extractor_quant.onnx")
    onnx_model_int8 = ONNXFasterRCNNWrapper(
        fe_onnx_path=os.path.join(quantized_fe_path),
        dh_onnx_path=dh_onnx_path,
        transform=model_fp32.transform,
        device=device
    )

    output_dir_onnx_int8 = os.path.join(
        cwd,
        "outputs",
        "frcnn",
        "onnx_val_int8"
    )

    if False:
        frcnn_utils.run_predictions_fasterrcnn(
            model=onnx_model_int8,
            data_loader=val_loader,
            device=device,
            dataset=dataset,
            output_folder=output_dir_onnx_int8,
            evaluate=False,
            num_batches=16,
            score_threshold=0.80
        )

    img = images[0:1]
    vutils.save_image(img[0],os.path.join(cwd, "outputs","frcnn","val_image.png"))

    fp32_feats=None
    int8_feats=None
    if False:
        fp32_feats = onnx_conv_outputs_from_batch(
            fe_onnx_path,
            img,
            transform=model_fp32.transform,
            device=device
        )

        int8_feats = onnx_conv_outputs_from_batch(
            quantized_fe_path,
            img,
            transform=model_fp32.transform,
            device=device
        )

        diffs = absolute_differences(fp32_feats, int8_feats)
        visualize_cnn_outputs(diffs, filename=os.path.join(cwd, "outputs", "frcnn", "feature_differences"))
        visualize_cnn_outputs(int8_feats, filename=os.path.join(cwd, "outputs", "frcnn", "feature_int8"))

        if False:
            frcnn_utils.run_predictions_fasterrcnn(
            model=onnx_model_int8,
            data_loader=calib_loader,
            device=device,
            dataset=dataset,
            output_folder=output_dir_onnx_int8,
            evaluate=False,
            num_batches=8,
            score_threshold=0.80
        )
        
    model_effi_fp32 = create_model(
        'tf_efficientdet_lite4',
        bench_task='predict',  # anchor decode + NMS
        pretrained=True
    )
    model_effi_fp32.to(device).eval()

    output_dir_effidet = os.path.join(os.path.dirname(__file__), "outputs", "effidet", "val_images")
    os.makedirs(output_dir_effidet, exist_ok=True)

    if False:
        frcnn_utils.run_predictions_efficientdet(
            model=model_effi_fp32,
            data_loader=val_loader,
            device=device,
            dataset=dataset,
            output_folder=output_dir_effidet,
            score_threshold=0.7,
            num_batches=5,
            target_size=(640,640)
        )

        model_path = os.path.join(model_dir, "tf_efficientdet_lite4.pth")
        torch.save(model_effi_fp32.state_dict(), model_path)
        print(f"Model saved at: {model_path}")

    import torch
    from torchvision import models, transforms
    from PIL import Image

    onnx_path = os.path.join(model_dir, "efficientnet_b0.onnx")
    if False:
        model_effi_fp32 = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model_effi_fp32.to(device).eval()

        imagenet_classes = models.EfficientNet_B0_Weights.IMAGENET1K_V1.meta["categories"]

        target_size = (224, 224)
        img_prepared = frcnn_utils.resize_and_pad(img[0].to(device), target_size).unsqueeze(0)

        with torch.no_grad():
            outputs = model_effi_fp32(img_prepared)
            probs = F.softmax(outputs, dim=1)
            conf, idx = torch.max(probs, dim=1)

        predicted_class = imagenet_classes[idx.item()]
        print(f"Predicted: {predicted_class}, confidence: {conf.item():.4f}")

        dummy_input = torch.randn(1, 3, 224, 224, device=device)

        torch.onnx.export(
            model_effi_fp32,
            dummy_input,
            onnx_path,
            input_names=['images'],
            output_names=['logits'],
            opset_version=17,
            dynamic_axes={'images': {0: 'batch_size'}, 'logits': {0: 'batch_size'}}
        )

        print(f"ONNX model exported to {onnx_path}")
    
    target_size = (224, 224) 
    quantized_model_path = os.path.join(model_dir, "efficientnet_b0_quant.onnx")
    if False:
        reader = SingleImageCalibrationReader(
            dataloader=calib_loader,
            input_name="images",
            target_size=target_size,
            device="cuda"
        )

        quantize_onnx_model_calibdl_int8(
            model_path=onnx_path,
            calib_data_loader=reader,
            quantized_model_path=quantized_model_path
        )
    
    if True:
        img2 = img[0].to(device)

        target_size = (224, 224)
        img_prepped = frcnn_utils.resize_and_pad(img2, target_size)

        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3,1,1)
        img_prepped = (img_prepped - mean) / std

        img_np = img_prepped.unsqueeze(0).cpu().numpy().astype(np.float32)

        sess = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name

        outputs = sess.run([output_name], {input_name: img_np})
        logits = torch.tensor(outputs[0])

        probs = torch.softmax(logits, dim=1)
        conf, idx = torch.max(probs, dim=1)

        imagenet_classes = models.EfficientNet_B0_Weights.IMAGENET1K_V1.meta["categories"]
        predicted_class = imagenet_classes[idx.item()]

        print(f"Predicted class: {predicted_class}, confidence: {conf.item():.4f}")
###################
        img_prepped = frcnn_utils.resize_and_pad(img2, target_size)

        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3,1,1)
        img_prepped = (img_prepped - mean) / std

        img_batch = img_prepped.unsqueeze(0)

        fp32_feats = onnx_conv_outputs_from_batch(
            onnx_path,
            images=img_batch,
            pattern = r".*",
            transform=None,
            device=device
        )

        logits = fp32_feats.pop("logits", None)
        max_vals = fp32_feats.pop("max", None)

        if len(fp32_feats) == 0:
            print("No CNN feature maps found to visualize.")
        else:
            visualize_cnn_outputs(
                fp32_feats,
                filename=os.path.join(cwd, "outputs", "frcnn", "fp32_features_effi")
            )

if __name__ == "__main__":
    main()