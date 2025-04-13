import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import logging
import cv2
import os
import logging
import re
import numpy as np

def save_image(image, filename="image", output_folder="outputs"):
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, filename + ".png")

    if isinstance(image, torch.Tensor):
        image = image.cpu().detach().numpy()
        image = np.transpose(image, (1, 2, 0))
        image = (image * 255).astype(np.uint8)

    cv2.imwrite(output_path, image)
    logging.info(f"Picture saved: {output_path}")


def get_dataloaders(dataset_class, root, transform=T.Compose([T.ToTensor()]), batch_size=32, train_split=0.8, seed=42, shuffle_train=True, **dataset_args):
    full_train_dataset = dataset_class(root=root, split="training", transforms=transform, **dataset_args)

    torch.manual_seed(seed)
    train_size = int(train_split * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle_train, collate_fn=lambda batch: tuple(zip(*batch))
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: tuple(zip(*batch))
    )

    logging.debug(f"Train and validation loaders are ready. They contain {train_size} and {val_size} images.")

    test_dataset = dataset_class(root=root, split="testing", transforms=transform, **dataset_args)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: tuple(zip(*batch))
    )

    test_dataset.class_to_idx = full_train_dataset.class_to_idx
    test_dataset.idx_to_class = full_train_dataset.idx_to_class

    logging.debug(f"Test loader is ready. It contains {len(test_dataset)} images.")

    return train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader, full_train_dataset.class_to_idx, full_train_dataset.idx_to_class

def video_to_dataloader(video_path, class_to_idx, idx_to_class, batch_size=32, transform=T.Compose([T.ToTensor()])):
    class VideoDataset(Dataset):
        def __init__(self, video_path, transform=None):
            self.video_path = video_path
            self.transform = transform
            self.frames = []
            self._extract_frames()

        def _extract_frames(self):
            cap = cv2.VideoCapture(self.video_path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.transform:
                    frame = self.transform(frame)
                self.frames.append(frame)
            cap.release()

        def __len__(self):
            return len(self.frames)

        def __getitem__(self, idx):
            return self.frames[idx], {"image_id": torch.tensor([idx])}
    
    dataset = VideoDataset(video_path, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: tuple(zip(*batch)))

    dataset.class_to_idx = class_to_idx
    dataset.idx_to_class = idx_to_class
    
    return dataset, dataloader

def create_video_from_images(directory, fps=30, output_dir="outputs"):
    files = [f for f in os.listdir(directory) if f.endswith('.png')]
    pattern = re.compile(r"batch(\d+)_img(\d+)\.png")
    
    try:
        files = sorted(files, key=lambda x: (int(pattern.match(x).group(1)), int(pattern.match(x).group(2))))
        logging.info("Files successfully sorted by batches and image numbers.")
    except Exception as e:
        logging.error(f"Error sorting by batches and images: {e}")
        files = sorted(files)
        logging.info("Files sorted by filename.")
    
    if not files:
        logging.warning("No PNG files found in the directory.")
        return
    
    first_image = cv2.imread(os.path.join(directory, files[0]))
    
    if first_image is None:
        logging.error(f"Failed to load the first image: {files[0]}")
        return

    height, width, _ = first_image.shape
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    video_output_path = os.path.join(output_dir, 'output_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_output = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))
    
    for file in files:
        img = cv2.imread(os.path.join(directory, file))
        if img is None:
            logging.warning(f"Skipping invalid image file: {file}")
            continue
        
        # Convert the image to BGR format if it is in any other format
        if img.shape[2] == 1:  # grayscale image (1 channel)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        video_output.write(img)
    
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass

    video_output.release()
    logging.info(f"Video creation completed: {video_output_path}")

def get_class_mapping(labels_dir=None, predefined_classes=None):
    class_list = []
    class_set = set()
    if predefined_classes:
        class_list = sorted(predefined_classes)
    else:
        if labels_dir and os.path.exists(labels_dir):
            for file in os.listdir(labels_dir):
                with open(os.path.join(labels_dir, file), "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_set.add(parts[0])
        class_list = sorted(class_set)

    class_to_idx = {cls: idx + 1 for idx, cls in enumerate(class_list)}
    idx_to_class = {idx + 1: cls for idx, cls in enumerate(class_list)}
    logging.info(f"Number of classes is {len(class_set)}.")
    logging.debug(f"Class list: {class_list}")

    return class_to_idx, idx_to_class

class KittiDataset(Dataset):
    def __init__(self, root, split="training", transforms=None):
        assert split in ["training", "testing"]
        self.root = root
        self.split = split
        self.transforms = transforms
        self.img_dir = os.path.join(root, split, "image_2")
        self.label_dir = os.path.join(root, split, "label_2")
        self.imgs = sorted(os.listdir(self.img_dir))
        if os.path.exists(self.label_dir):
            self.labels = sorted(os.listdir(self.label_dir))
            self.class_to_idx, self.idx_to_class = get_class_mapping(self.label_dir)
        else:
            self.labels = None
            self.class_to_idx, self.idx_to_class = {}, {}
        logging.info(f"Found {len(self.imgs)} images in {split} set.")
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.labels is not None:
            label_path = os.path.join(self.label_dir, self.labels[idx])
            boxes = []
            labels = []
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 8:
                        left, top, right, bottom = map(float, parts[4:8])
                        boxes.append([left, top, right, bottom])
                        labels.append(self.class_to_idx.get(parts[0], 0))
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}
        else:
            target = {"image_id": torch.tensor([idx])}
        if self.transforms:
            img = self.transforms(img)
        return img, target
    def __len__(self):
        return len(self.imgs)
