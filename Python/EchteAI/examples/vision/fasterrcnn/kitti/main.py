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

    train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader, class_to_idx, idx_to_class = dl.get_dataloaders(dl.KittiDataset,root,transform)

    num_epochs = 16
    model = frcnn.setup_fasterrcnn(train_dataset)
    model.to(device)
    model = frcnn.train_fasterrcnn(model, train_loader, val_loader, device, num_epochs)
    model.eval()

    val_out = os.path.join("outputs", "predictions", "validation")
    test_out = os.path.join("outputs", "predictions", "test")
    #frcnn.run_predictions_fasterrcnn(model, val_loader, device, val_dataset.dataset if hasattr(val_dataset, "dataset") else val_dataset, val_out, evaluate=True, num_batches=2)
    #frcnn.run_predictions_fasterrcnn(model, test_loader, device, test_dataset, test_out, evaluate=False, num_batches=2)

    device = "cpu"
    model_quantized = frcnn.quantize_fasterrcnn(model, train_loader)
    model_quantized = frcnn.quantize_dynamic(model_quantized)
    print(model_quantized)
    val_out_qint8_static = os.path.join("outputs", "qint8_static", "validation")
    #frcnn.run_predictions_fasterrcnn(model_quantized, val_loader, device, val_dataset.dataset if hasattr(val_dataset, "dataset") else val_dataset, val_out_qint8_static, evaluate=False, num_batches=2)

    video_set, video_loader = dl.video_to_dataloader("car_video.mp4", class_to_idx, idx_to_class)
    video_out_qint8_static = os.path.join("outputs", "qint8_static", "video")
    frcnn.run_predictions_fasterrcnn(model_quantized, video_loader, device, video_set.dataset if hasattr(video_set, "dataset") else video_set, video_out_qint8_static, evaluate=False, num_batches=25)
    dl.create_video_from_images(video_out_qint8_static)

if __name__ == "__main__":
    main()
