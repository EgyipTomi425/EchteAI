import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import logging
import cv2
import os
import logging

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

    logging.debug(f"Test loader is ready. It contains {len(test_dataset)} images.")

    return train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader

def get_class_mapping(labels_dir=None, predefined_classes=None):
    if predefined_classes:
        class_list = sorted(predefined_classes)
    else:
        class_set = set()
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
            self.class_to_idx, self.idx_to_class = get_class_mapping(root, split)
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
