import EchteAI.data.preprocessing as pp
import EchteAI.data.dataloaders as dl
import EchteAI.models.vision.fasterrcnn as frcnn
import logging
import torchvision.transforms as T
import torch

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    root = "./downloads"
    image_zip_url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip"
    label_zip_url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip"
    pp.download_data([image_zip_url, label_zip_url], download_dir=root)
    transform = T.Compose([T.ToTensor()])

    train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader = dl.get_dataloaders(dl.KittiDataset,root,transform)

    num_epochs = 8
    model = frcnn.setup_fasterrcnn(train_dataset)
    model.to(device)
    model = frcnn.train_fasterrcnn(model, train_loader, val_loader, device, num_epochs)
    model.eval()

if __name__ == "__main__":
    main()
