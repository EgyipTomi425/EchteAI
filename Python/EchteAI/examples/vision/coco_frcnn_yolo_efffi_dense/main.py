# https://cocodataset.org/#download
# Dataset 2017

import warnings

from EchteAI.models.vision.models.onnx_frcnn import quantize_feature_extractor
warnings.filterwarnings("ignore")

import logging
import os
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split

import EchteAI.data.dataloaders as dl
from EchteAI.models.vision.models.fasterrcnn_split import ONNXFasterRCNNWrapper, split_save_frcnn
import EchteAI.models.vision.models.fasterrcnn_utils as frcnn_utils

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

    batch_size = 4

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

    onnx_model_int8 = ONNXFasterRCNNWrapper(
        fe_onnx_path=os.path.join(model_dir, "feature_extractor_quant.onnx"),
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

    if True:
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

if __name__ == "__main__":
    main()