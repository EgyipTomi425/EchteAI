# https://cocodataset.org/#download
# Dataset 2017

import logging
import os
import json
from typing import List, Tuple
import shutil

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

import EchteAI.data.dataloaders as dl
import EchteAI.models.vision.models.fasterrcnn_utils as frcnn_utils
import EchteAI.models.vision.visualization as vis


class ImageFolderDataset(Dataset):
    """Dataset that returns PIL images."""
    def __init__(self, root: str):
        self.root = root
        self.files = sorted([f for f in os.listdir(root) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        # class mapping
        self.class_to_idx = {}
        self.idx_to_class = {}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx) -> Tuple:
        path = os.path.join(self.root, self.files[idx])
        import PIL.Image as Image
        img = Image.open(path).convert("RGB")
        target = {"image_id": torch.tensor([idx])}
        return img, target


def pil_to_tensor_collate_fn(batch):
    """Convert PIL images to tensors."""
    transform = T.ToTensor()
    images = []
    targets = []
    for img, target in batch:
        images.append(transform(img))
        targets.append(target)
    return images, targets


def pil_collate_fn(batch):
    """Collate PIL images without conversion to tensors (for calibration)."""
    images = []
    targets = []
    for img, target in batch:
        images.append(img)
        targets.append(target)
    return images, targets


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    cwd = os.path.dirname(__file__)
    image_dir = os.path.join(cwd, "downloads", "val2017")
    annotations_path = os.path.join(cwd, "downloads", "annotations", "instances_val2017.json")
    output_dir_fp32 = os.path.join(cwd, "outputs", "frcnn", "val_fp32")

    if not os.path.exists(image_dir):
        logging.error(f"Input image folder not found: {image_dir}")
        return

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    batch_size = 4

    dataset = ImageFolderDataset(image_dir)
    
    idx_to_class = {0: '__background__'}
    if os.path.exists(annotations_path):
        with open(annotations_path, 'r') as f:
            coco_data = json.load(f)
            cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}
            idx_to_class.update({cat_id: name for cat_id, name in cat_id_to_name.items()})
        logging.info(f"Loaded {len(idx_to_class) - 1} category names from {annotations_path}")
    else:
        logging.warning(f"Annotations file not found: {annotations_path}")
    
    dataset.idx_to_class = idx_to_class
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=pil_to_tensor_collate_fn)

    logging.info("Loading FP32 model...")
    model_fp32 = frcnn_utils.setup_fasterrcnn(dataset=None, backbone="resnet50")
    model_fp32.to(device).eval()
    logging.info("Running predictions on original FP32 model...")
    frcnn_utils.run_predictions_fasterrcnn(model=model_fp32, data_loader=data_loader, device=device, dataset=dataset, output_folder=output_dir_fp32, evaluate=False, num_batches=3, score_threshold=0.85)


if __name__ == "__main__":
    main()