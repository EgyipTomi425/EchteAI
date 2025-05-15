import EchteAI.data.preprocessing as pp
import EchteAI.data.dataloaders as dl
import EchteAI.models.vision.fasterrcnn as frcnn
import logging
import torchvision.transforms as T
import torch
import os
import onnx
import torch.nn.functional as F

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

file_handler = logging.FileHandler('output.log', mode='a', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)

# Azonnali flush minden logolás után
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
file_handler.flush = lambda: None  # Kikapcsolja a pufferelést

logger = logging.getLogger()
logger.addHandler(file_handler)

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

    device = "cpu"
    val_out = os.path.join("outputs", "predictions", "validation")
    test_out = os.path.join("outputs", "predictions", "test")
    #frcnn.run_predictions_fasterrcnn(model, val_loader, device, val_dataset.dataset if hasattr(val_dataset, "dataset") else val_dataset, val_out, evaluate=True, num_batches=3)
    #frcnn.run_predictions_fasterrcnn(model, test_loader, device, test_dataset, test_out, evaluate=False, num_batches=2)

    device = "cpu"
    #model_quantized = frcnn.quantize_fasterrcnn(model, train_loader)
    #model_quantized = frcnn.quantize_dynamic(model_quantized)
    #print(model_quantized)
    val_out_qint8_static = os.path.join("outputs", "qint8_static", "validation")
    #frcnn.run_predictions_fasterrcnn(model_quantized, val_loader, device, val_dataset.dataset if hasattr(val_dataset, "dataset") else val_dataset, val_out_qint8_static, evaluate=True, num_batches=3)

    #video_set, video_loader = dl.video_to_dataloader("car_video.mp4", class_to_idx, idx_to_class)
    #video_out_qint8_static = os.path.join("outputs", "qint8_static", "video")
    #frcnn.run_predictions_fasterrcnn(model_quantized, video_loader, device, video_set.dataset if hasattr(video_set, "dataset") else video_set, video_out_qint8_static, evaluate=False, num_batches=3)
    #frcnn.compare_models_visual(model, model_quantized, video_loader, device, video_set.dataset if hasattr(video_set, "dataset") else video_set, video_out_qint8_static, num_batches=-1)
    #dl.create_video_from_images(video_out_qint8_static)

    #torch.save(model_quantized, "quantized_model.pth")



    images, _ = next(iter(val_loader))
    images = torch.stack([F.interpolate(img.unsqueeze(0), size=(375, 1242), mode="bilinear", align_corners=False).squeeze(0) for img in images], dim=0).to(device)
    images = images[:32]
    #torch.onnx.export(model.backbone, images[:15], "./outputs/model.onnx")
    first_batch = next(iter(val_loader))
    first_image = first_batch[0][13].to(device)
    dl.save_image(first_image)

    frcnn.split_save_frcnn(model, images[:2], device)
    model_onnx_fp32 = frcnn.ONNXFasterRCNNWrapper("feature_extractor.onnx", "detector_head.onnx", "cpu")
    val_out_onnx_fp32 = os.path.join("outputs", "onnx", "fp32")
    frcnn.run_predictions_fasterrcnn(model_onnx_fp32, val_loader, device, val_dataset.dataset if hasattr(val_dataset, "dataset") else val_dataset, val_out_onnx_fp32, evaluate=False, num_batches=3, batch_size=2)

    frcnn.quantize_onnx_static(
        onnx_model_path="feature_extractor.onnx",
        data_loader=train_loader,
        input_shape=(2, 3, 375, 1242),
        num_batches=8
    )
    model_onnx_int8 = frcnn.ONNXFasterRCNNWrapper("feature_extractor_int8.onnx", "detector_head.onnx", "cpu")
    val_out_onnx_int8 = os.path.join("outputs", "onnx", "int8")
    frcnn.run_predictions_fasterrcnn(model_onnx_int8, val_loader, device, val_dataset.dataset if hasattr(val_dataset, "dataset") else val_dataset, val_out_onnx_int8, evaluate=False, num_batches=3, batch_size=2)

    #outputs1 = frcnn.backbone_cnn_layers_outputs(model_quantized, first_image)
    #outputs2 = frcnn.backbone_cnn_layers_outputs(model, first_image)
    #outputs_diffs = frcnn.absolute_differences(outputs1, outputs2)
    #outputs_percentage_diffs = frcnn.percentage_differences(outputs1, outputs2)
    #frcnn.visualize_cnn_outputs(outputs1)
    #frcnn.visualize_cnn_outputs(outputs_diffs, filename="activation_difference_heatmap")
    #frcnn.visualize_cnn_outputs(outputs_diffs, filename="activation_difference_heatmap_layer1", layer=1)
    #frcnn.visualize_cnn_outputs(outputs_percentage_diffs, filename="activation_difference_heatmap_percentage")
    #frcnn.visualize_cnn_outputs(outputs_percentage_diffs, filename="activation_difference_heatmap_percentage_layer1")
    #frcnn.fit_and_plot_distribution(outputs1, outputs_diffs, layer=1)
    #frcnn.fit_and_plot_distribution(outputs1, outputs_percentage_diffs, layer=1, filename="distribution_fit_percentage")

if __name__ == "__main__":
    main()
