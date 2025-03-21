from EchteAI.data import preprocessing, dataloaders

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    root = "./downloads"
    image_zip_url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip"
    label_zip_url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip"
    download_data([image_zip_url, label_zip_url], download_dir=root)
    transform = T.Compose([T.ToTensor()])

    train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader = get_dataloaders(KittiDataset,root,transform)

if __name__ == "main":
    main()
