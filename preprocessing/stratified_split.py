import os
import shutil
import numpy as np
import yaml
from pathlib import Path
from typing import Tuple, Dict, List
from skmultilearn.model_selection import IterativeStratification
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def stratified_split(
    yaml_path: str,
    output_path: str,
    split_ratio: Tuple[float, float, float] = (0.7, 0.2, 0.1),
    max_workers: int = 8
):
    """
    Function ini untuk melakukan pembagian data 
    dengan metode stratified agar setiap split 
    memiliki distribusi kelas yang seimbang

    Args:
        yaml_path (str): _description_
        output_path (str): _description_
        split_ratio (Tuple[float, float, float], optional): _description_. Defaults to (0.7, 0.2, 0.1).
        max_workers (int, optional): _description_. Defaults to 8.
    """

    if np.isclose(sum(split_ratio), 1.0) == False:
        print("Invalid Ratio tolong cek kembali")
        return None
        

    y_path = Path(yaml_path).resolve()
    out_path = Path(output_path)
    source_path = y_path.parent
    source_img = source_path/"images"
    source_label = source_path/"labels"


    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        num_classes = config['nc']
        class_names = config['names']

    except Exception as err:
        print("Error ketika read yaml file: ", err)
        return None
    
    if source_img.is_dir() == False or source_label.is_dir() == False:
        split_dirs = ['train', 'valid', 'test']
        found_splits = []
        
        for split in split_dirs:
            split_img = source_path / split / 'images'
            split_lbl = source_path / split / 'labels'
            if split_img.is_dir() and split_lbl.is_dir():
                found_splits.append(split)
        
        if found_splits == False:
            print("source image/label tidak ditemukan!, tolong cek kembali datanya!")
            return None
        
        temp_combined = source_path / '_temp_combined'
        temp_images = temp_combined / 'images'
        temp_labels = temp_combined / 'labels'
        temp_images.mkdir(parents=True, exist_ok=True)
        temp_labels.mkdir(parents=True, exist_ok=True)
        
        total_copied = 0
        for split in found_splits:
            split_img = source_path / split / 'images'
            split_lbl = source_path / split / 'labels'
            
            for img_file in split_img.iterdir():
                if img_file.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}:
                    shutil.copy2(img_file, temp_images / img_file.name)
                    total_copied += 1
            
            for lbl_file in split_lbl.iterdir():
                if lbl_file.suffix.lower() == '.txt':
                    shutil.copy2(lbl_file, temp_labels / lbl_file.name)

        source_img = temp_images
        source_label = temp_labels
        
        print(f"Temp Images: {source_img}")
        print(f"Temp Labels: {source_label}")
    
    ext = {'.png', '.jpg', '.jpeg'}
    
    file_name = []
    for f in source_img.iterdir():
        if f.suffix.lower() in ext:
            file_name.append(f.name)

    image_files = sorted(file_name)

    print("ditemukan gambar sebanyak: ", len(image_files))

    x = np.array(image_files).reshape(-1, 1)
    y = label_matrix(source_label, image_files, num_classes, max_workers)

    x_train, x_val, x_test = perform_stratified_split(x, y, split_ratio)

    print(f"Split:")
    print(f"Train: {len(x_train)} images ({len(x_train)/len(x) * 100:.1f}%)")
    print(f"Valid: {len(x_val)} images ({len(x_val)/len(x) * 100:.1f}%)")
    print(f"Test:  {len(x_test)} images ({len(x_test)/len(x) * 100:.1f}%)")

    splits = {
        'train': x_train,
        'valid': x_val,
        'test': x_test,
    }

    for split_name, file_list in splits.items():
        if len(file_list) == 0:
            continue
        
        print("Processing set: ", split_name)
        output_img_path = out_path / split_name / 'images'
        output_lbl_path = out_path / split_name / 'labels'

        copy_files(file_list, source_img, source_label, output_img_path, output_lbl_path)

    create_yaml(out_path, num_classes, class_names)
    print("\nStratified Split Selesai!")
    print("jumlah image: ", len(x))
    print(f"Output path: {out_path.resolve()}")
    print(f"Split ratios: Train={split_ratio[0]:.0%}, Val={split_ratio[1]:.0%}, Test={split_ratio[2]:.0%}")


def label_matrix(
    source_label: Path,
    image_files: List[str],
    num_classes: int,
    max_workers: int = 8
):

    label_matrix = np.zeros((len(image_files), num_classes), dtype=np.int8)
    image_to_idx = {}

    for i, name in enumerate(image_files):
        image_to_idx[name] = i

    label_files = list(source_label.glob('*.txt'))

    matrixes = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for label in label_files:
            x = ex.submit(process_label, label, image_to_idx, num_classes)
            matrixes.append(x)

        for matrix in tqdm(as_completed(matrixes), total=len(matrixes)):
            idx, vector = matrix.result()
            if idx != None:
                label_matrix[idx] = vector

    return label_matrix



def process_label(label_path: Path, image_to_idx: Dict[str, int], num_classes: int):
    base_name = label_path.stem

    for ext in ['.jpg', '.jpeg', '.png']:
        img_name = base_name + ext

        if img_name in image_to_idx:
            idx = image_to_idx[img_name]
            vector = parse_label(label_path, num_classes)
            return idx, vector
        
    return None, None

def parse_label(label_path: Path, num_classes: int):
    class_vector = np.zeros(num_classes, dtype=np.int8)
    
    try:
        with open(label_path, 'r') as label_file:
            for line in label_file:
                parts = line.strip().split()
                if parts == None or parts == "":
                    raise Exception

                class_id = int(parts[0])
                if 0 <= class_id < num_classes:
                    class_vector[class_id] = 1
    except Exception as err:
        print("Unable to parse label: ", err)
        return None
    
    return class_vector


def perform_stratified_split(x: np.ndarray, y: np.ndarray, split_ratios: Tuple[float, float, float]):
    
    train_ratio, val_ratio, test_ratio = split_ratios
    
    stratifier = IterativeStratification(
        n_splits=2,
        order=1,
        sample_distribution_per_fold=[test_ratio + val_ratio, train_ratio]
    )
    train_indices, rest_indices = next(stratifier.split(x, y))
    
    x_train = x[train_indices]
    x_rest = x[rest_indices]
    y_rest = y[rest_indices]
    
    if val_ratio + test_ratio > 0:
        test_vs_val_ratio = test_ratio / (test_ratio + val_ratio)
        stratifier_rest = IterativeStratification(
            n_splits=2,
            order=1,
            sample_distribution_per_fold=[test_vs_val_ratio, 1.0 - test_vs_val_ratio]
        )
        test_indices, val_indices = next(stratifier_rest.split(x_rest, y_rest))
        x_val = x_rest[val_indices]
        x_test = x_rest[test_indices]
    else:
        x_val = np.array([])
        x_test = np.array([])
    
    return x_train, x_val, x_test


def copy_files(
    file_list: np.ndarray,
    source_images: Path,
    source_labels: Path,
    output_img_path: Path,
    output_lbl_path: Path,
    max_workers: int = 8
):
    output_img_path.mkdir(parents=True, exist_ok=True)
    output_lbl_path.mkdir(parents=True, exist_ok=True)


    filenames = []
    for f in file_list:
        filenames.append(f[0])
    
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        matrixes = []
        for f in filenames:
            matrix = ex.submit(copy_file_pair, f, source_images, source_labels, output_img_path, output_lbl_path)
            matrixes.append(matrix)
        
        list(tqdm(as_completed(matrixes),total=len(matrixes)))


def copy_file_pair(
    filename: str,
    source_images: Path,
    source_labels: Path,
    output_img_path: Path,
    output_lbl_path: Path
):
    base_name = Path(filename).stem
    
    src_img = source_images / filename
    dst_img = output_img_path / filename
    if src_img.exists():
        shutil.copy2(src_img, dst_img)
    
    src_lbl = source_labels / f"{base_name}.txt"
    dst_lbl = output_lbl_path / f"{base_name}.txt"
    if src_lbl.exists():
        shutil.copy2(src_lbl, dst_lbl)


def create_yaml(output_path: Path, num_classes: int, class_names: List[str]):
    yaml_path = output_path / 'data.yaml'
    
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(f"train: train/images\n")
        f.write(f"val: valid/images\n")
        f.write(f"test: test/images\n")
        f.write(f"\n")
        f.write(f"nc: {num_classes}\n")
        f.write(f"names: {class_names}\n")
    
    print(f"'data.yaml' dibuat di '{yaml_path}'")

