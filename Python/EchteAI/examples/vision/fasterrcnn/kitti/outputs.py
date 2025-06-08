import os
import shutil

# Elérési utak
src_img_dir = "downloads/yolo_dataset/images/val"
src_lbl_dir = "downloads/yolo_dataset/labels/val"
dst_img_dir = "downloads/yolo_dataset/images/test"
dst_lbl_dir = "downloads/yolo_dataset/labels/test"

# Mappák létrehozása, ha nem léteznek
os.makedirs(dst_img_dir, exist_ok=True)
os.makedirs(dst_lbl_dir, exist_ok=True)

# Képfájlok listázása (pl. .png vagy .jpg)
image_files = sorted([f for f in os.listdir(src_img_dir) if f.endswith((".png", ".jpg"))])[:100]

# Másolás
for f in image_files:
    # Fájlnevek
    base = os.path.splitext(f)[0]
    img_src = os.path.join(src_img_dir, f)
    lbl_src = os.path.join(src_lbl_dir, base + ".txt")
    
    img_dst = os.path.join(dst_img_dir, f)
    lbl_dst = os.path.join(dst_lbl_dir, base + ".txt")

    # Másolás
    shutil.copy(img_src, img_dst)
    if os.path.exists(lbl_src):
        shutil.copy(lbl_src, lbl_dst)

