import os
import shutil
import random
from pathlib import Path


def split_dataset(source_dir, benign_dir, trojan_dir, split_ratio=0.75):
    Path(benign_dir).mkdir(parents=True, exist_ok=True)
    Path(trojan_dir).mkdir(parents=True, exist_ok=True)

    for label in os.listdir(source_dir):
        label_path = os.path.join(source_dir, label)

        if not os.path.isdir(label_path):
            continue

        benign_label_dir = os.path.join(benign_dir, label)
        trojan_label_dir = os.path.join(trojan_dir, label)
        os.makedirs(benign_label_dir, exist_ok=True)
        os.makedirs(trojan_label_dir, exist_ok=True)

        files = [f for f in os.listdir(label_path) if os.path.isfile(os.path.join(label_path, f))]
        random.shuffle(files)
        split_point = int(len(files) * split_ratio)
        for file in files[:split_point]:
            src = os.path.join(label_path, file)
            dst = os.path.join(benign_label_dir, file)
            shutil.copy2(src, dst)
        for file in files[split_point:]:
            src = os.path.join(label_path, file)
            dst = os.path.join(trojan_label_dir, file)
            shutil.copy2(src, dst)

    print(f"数据集划分完成! Benign_Data: {benign_dir}, Trojan_Data: {trojan_dir}")


if __name__ == "__main__":
    source_directory = "D://Data//Voxceleb"  # 替换为你的数据集路径
    benign_directory = "Benign_Data"
    trojan_directory = "Trojan_Data"

    split_dataset(source_directory, benign_directory, trojan_directory)