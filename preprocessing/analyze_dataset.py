from pathlib import Path
import yaml
from collections import defaultdict, Counter

def analyze_dataset(data_yaml_path: str):
    """
    Function untuk menganalisis total dataset dari file yaml
    """
    yaml_path = Path(data_yaml_path).resolve()

    if yaml_path.exists() == False:
        print("File yaml tidak temukan, tolong cek ulang kembali!")
        return None
    
    try:
        data = yaml.safe_load(yaml_path.read_text())
    except Exception as err:
        print("Err:", err)
        return None
    
    if "names" in data == False:
        print("Class name tidak temukan di file yaml, tolong cek file yaml anda!")
        return None
    
    class_names = data["names"]
    class_count = len(class_names)

    print(f"Jumlah Class: {class_count}")
    print(f"Nama Class: {', '.join(class_names)}")

    base_path = yaml_path.parent

    total_images = 0
    total_objects = 0
    total_counts = defaultdict(int)

    for split in ["train", "val", "test"]:
        if split in data == False:
            continue

        split_path = data[split]
        if split_path == None:
            continue

        image_path = (base_path / split_path).resolve()
        label_path = image_path.parent / "labels"

        split_res = analyze_split(split, image_path, label_path, class_names)
        if split_res == None:
            continue

        total_images += split_res["total_images"]
        total_objects += split_res["total_objects"]

        for name, count in split_res["class_distribution"].items():
            total_counts[name] += count

    print(f"\nTotal Gambar di Semua Split: {total_images}")

    print("\n Distribusi Objek Gabungan:")
    
    if total_objects == 0:
        print("Tidak ada objek ditemukan.")
        return None

    for i, (name, count) in enumerate(sorted(total_counts.items(), key=lambda x: x[1], reverse=True), start=1):
        percent = (count / total_objects) * 100
        print(f"{i:>2}. {name:<15}: {count} jumlah | {percent:.2f}%")

    print(f"Total Semua Objek: {total_objects}")

    return {
        "total_images": total_images,
        "total_objects": total_objects,
        "class_distribution": dict(total_counts),
    }


def analyze_split(split_name: str, image_path: Path, label_path: Path, class_names: list[str]):
    """
    Function ini untuk melakukan analisis jumlah split dataset (train/val/test) dari images dan label
    """
    if image_path.is_dir() == False or label_path.is_dir() == False:
        print("Ada lokasi split yang tidak ditemukan, tolong cek kembali path split anda!")
        return None
    
    image_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_files.extend(image_path.glob(ext))

    image_names = set()
    for i in image_files:
        image_names.add(i.stem)

    total_images = len(image_files)
    class_counter = Counter()
    total_objects = 0

    valid_labels = []
    for l in label_path.glob("*.txt"):
        if l.stem in image_names:
            valid_labels.append(l)

    for vl in valid_labels:
        try:
            with vl.open("r") as label:
                for line in label:
                    stripped = line.strip()
                    if stripped == "":
                        continue

                    parts = stripped.split()
                    class_id_str = parts[0]
                    
                    if class_id_str.isdigit() == False:
                        continue
                    
                    class_id = int(class_id_str)
                    if class_id >= 0 and class_id < len(class_names):
                        class_counter.update([class_names[class_id]])
                        total_objects += 1
        except Exception as err:
            print("Error:", err)
            continue
        
    if total_objects == 0:
        print("Objects tidak ditemukan!")
        return None
    
    print(f"\n Jumlah Objects pada split '{split_name}':")
    for i, (name, count) in enumerate(sorted(class_counter.items(), key=lambda x: x[1], reverse=True), start=1):
        percent = (count / total_objects) * 100
        print(f"{i:>2}. {name:<15}: {count} jumlah | {percent:.2f}%")

    return {
        "split": split_name,
        "total_images": total_images,
        "total_labels": len(valid_labels),
        "total_objects": total_objects,
        "class_distribution": dict(class_counter),
    }
