import EchteAI.data.preprocessing as pp
import EchteAI.data.dataloaders as dl
import EchteAI.models.vision.fasterrcnn as frcnn
import logging
import torchvision.transforms as T
import torch
import os

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    root = "./downloads"
    image_zip_url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip"
    label_zip_url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip"
    pp.download_data([image_zip_url, label_zip_url], download_dir=root)
    transform = T.Compose([T.ToTensor()])

    train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader = dl.get_dataloaders(dl.KittiDataset,root,transform)

    num_epochs = 16
    model = frcnn.setup_fasterrcnn(train_dataset)
    model.to(device)
    model = frcnn.train_fasterrcnn(model, train_loader, val_loader, device, num_epochs)
    model.eval()

    val_out = os.path.join("outputs", "predictions", "validation")
    test_out = os.path.join("outputs", "predictions", "test")
    frcnn.run_predictions_fasterrcnn(model, val_loader, device, val_dataset.dataset if hasattr(val_dataset, "dataset") else val_dataset, val_out, evaluate=True, num_batches=2)
    frcnn.run_predictions_fasterrcnn(model, test_loader, device, test_dataset, test_out, evaluate=False, num_batches=2)


if __name__ == "__main__":
    main()
