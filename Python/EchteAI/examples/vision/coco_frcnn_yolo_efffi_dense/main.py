# https://cocodataset.org/#download
# Dataset 2017

import logging
import os
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split

import EchteAI.data.dataloaders as dl
import EchteAI.models.vision.models.fasterrcnn_utils as frcnn_utils

torch.manual_seed(42)

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

    if not os.path.exists(image_dir):
        logging.error(f"Input image folder not found: {image_dir}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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


if __name__ == "__main__":
    main()