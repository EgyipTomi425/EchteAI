import os
import requests
import zipfile
import tarfile
import logging
from tqdm import tqdm

def download_data(urls, download_dir="./downloads"):
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    else:
        logging.info(f"Download directory already exists. Skipping downloads.")
        return -1
    
    for url in urls:
        filename = os.path.join(download_dir, url.split("/")[-1])

        try:
            logging.info(f"Downloading: {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            
            with open(filename, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading", miniters=1, disable=False) as pbar:
                    for chunk in response.iter_content(chunk_size=1048576):
                        f.write(chunk)
                        pbar.update(len(chunk))

            logging.info(f"Download complete: {filename}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Error during download {url}: {e}")
            continue

        try:
            logging.info(f"Extracting: {filename}")
            if filename.endswith('.zip'):
                with zipfile.ZipFile(filename, 'r') as zip_ref:
                    total_files = len(zip_ref.namelist())
                    with tqdm(total=total_files, desc="Extracting ZIP", disable=False) as pbar:
                        for file in zip_ref.namelist():
                            zip_ref.extract(file, download_dir)
                            pbar.update(1)
            elif filename.endswith(('.tar', '.tar.gz', '.tgz')):
                with tarfile.open(filename, 'r:*') as tar_ref:
                    total_files = len(tar_ref.getnames())
                    with tqdm(total=total_files, desc="Extracting TAR", disable=False) as pbar:
                        for file in tar_ref.getnames():
                            tar_ref.extract(file, download_dir)
                            pbar.update(1)

            logging.info(f"Extraction complete: {filename}")
        except (zipfile.BadZipFile, tarfile.TarError) as e:
            logging.error(f"Error during extraction of {filename}: {e}")
            continue
        finally:
            if os.path.exists(filename):
                os.remove(filename)
                logging.info(f"Downloaded file removed: {filename}")
    return 0
