from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import imagehash
import yaml
import shutil
from collections import defaultdict
from typing import Dict, List, Tuple, Set

def remove_duplicates(
    yaml_path: str, 
    max_workers: int = 8,
    similiarity_threshold: int = 15,
    exact_threshold: int = 3,
    preview_only = True,
    use_full_scan: bool = False
):
    """
    Function ini untuk mendeteksi seluruh gambar duplikat 
    dari dataset termasuk splitnya (train, val, dan test) 
    berdasarkan pendeketan multi-hash
    Args:
        yaml_file_path: Path ke file data.yaml
        max_workers: Jumlah worker untuk parallel processing
        similarity_threshold: Threshold untuk duplikat dengan augmentasi (0-60)
                            - 10-15: Augmentasi ringan (flip, rotate, crop kecil)
                            - 16-25: Augmentasi medium (brightness, contrast, blur)
                            - >25: Augmentasi agresif
        exact_threshold: Threshold untuk duplikat exact/hampir identik (0-10)
        preview_only: Jika True, hanya tampilkan preview tanpa menghapus
        use_full_scan: Jika True, compare semua gambar (O(n²), lambat tapi 100% akurat)
                       Jika False, gunakan bucketing (cepat, mungkin miss beberapa edge cases)
    """

    path = Path(yaml_path).resolve()

    try:
        data = yaml.safe_load(path.read_text())
    except Exception as err:
        print("Error reading yaml: ", err)
        return None
    
    base_path = path.parent
    all_images = []

    for split in ("train", "val", "test"):
        if split in data and data[split]:
            image_path = (base_path/data[split]).resolve()
            if image_path.is_dir():
                all_images.extend(image_path.glob("*.jpg"))
                all_images.extend(image_path.glob("*.jpeg"))
                all_images.extend(image_path.glob("*.png"))

    total_images = len(all_images)
    if total_images == 0:
        print("foto tidak ditemukan, tolong cek data kembali!")

    print("total foto ditemukan: ", total_images)

    hash_data = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        res = list(ex.map(process_images, all_images))
        for r in res:
            if r != None:
                hash_data.append(r)

    dups_map: Dict[Path, Set[Path]] = defaultdict(set)
    exact_dups = set() # buat hapus file yang duplikat atau sangat mirip
    similar_dups = set() # buat hapus file yang memiliki augmentasi file dengan beda filename

    checked = set()
    total_checked = 0

    if use_full_scan == True:
        print("Mode Full Scan Aktif, ini bakal lama yaaa")
        max_comparisons = len(hash_data) * (len(hash_data) - 1) // 2
        print(f"komparasi yang akan dilakukan sebanyak: ~{max_comparisons}")

        progress_interval = max_comparisons // 20

        for i, data1 in enumerate(hash_data):
            path1 = data1[4]
            
            if path1 in exact_dups or path1 in similar_dups:
                continue
            
            for data2 in hash_data[i+1:]:
                path2 = data2[4]
                
                if path2 in exact_dups or path2 in similar_dups:
                    continue
                
                total_checked += 1

                if progress_interval > 0 and total_checked % progress_interval == 0:
                    progress = (total_checked / max_comparisons) * 100
                    print(f"Progress: {progress:.1f}% ({total_checked}/{max_comparisons})")
                
                score = cal_similarity_score(data1, data2)

                if score <= exact_threshold:
                    dups_map[path1].add(path2)
                    exact_dups.add(path2)
                    print(f"Foto duplikat/mirip banget dengan nilai ({score}): {path2.name} == {path1.name}")

                elif score <= similiarity_threshold:
                    dups_map[path1].add(path2)
                    similar_dups.add(path2)
                    print(f"Foto augmentasi dengan nilai ({score}): {path2.name} == {path1.name}")

    else :
        hash_map: Dict[str, List] = defaultdict(list)
        for data in hash_data:
            ahash = data[0]

            hash_key = str(ahash)[:4]
            hash_map[hash_key].append(data)

        for value in hash_map.values():
            for i, data1 in enumerate(value):
                path1 = data1[4]

                for data2 in value[i+1:]:
                    path2 = data2[4]

                    if path2 in exact_dups or path2 in similar_dups:
                        continue

                    emotion_set = tuple(sorted([str(path1), str(path2)]))
                    if emotion_set in checked:
                        continue

                    checked.add(emotion_set)

                    total_checked += 1
                    score = cal_similarity_score(data1, data2)

                    if score <= exact_threshold:
                        dups_map[path1].add(path2)
                        exact_dups.add(path2)
                        print(f"Foto duplikat/mirip banget dengan nilai ({score}): {path2.name} == {path1.name}")

                    elif score <= similiarity_threshold:
                        dups_map[path1].add(path2)
                        similar_dups.add(path2)
                        print(f"Foto augmentasi dengan nilai ({score}): {path2.name} == {path1.name}")

        bucket_keys = sorted(hash_map.keys())
        cross_checked = 0
        
        for i, key1 in enumerate(bucket_keys):
            for key2 in bucket_keys[i+1:min(i+11, len(bucket_keys))]:  # Max 10 bucket terdekat
                try:
                    if abs(int(key1, 16) - int(key2, 16)) > 0x2000:
                        continue
                except ValueError:
                    continue
                
                for data1 in hash_map[key1]:
                    path1 = data1[4]
                    if path1 in exact_dups or path1 in similar_dups:
                        continue
                    
                    for data2 in hash_map[key2]:
                        path2 = data2[4]
                        if path2 in exact_dups or path2 in similar_dups:
                            continue
                        
                        emotion_set = tuple(sorted([str(path1), str(path2)]))
                        if emotion_set in checked:
                            continue
                        
                        checked.add(emotion_set)
                        total_checked += 1
                        cross_checked += 1
                        score = cal_similarity_score(data1, data2)
                        
                        if score <= exact_threshold:
                            dups_map[path1].add(path2)
                            exact_dups.add(path2)
                            print(f"Foto duplikat/mirip banget dengan nilai ({score}): {path2.name} == {path1.name}")
                        
                        elif score <= similiarity_threshold:
                            dups_map[path1].add(path2)
                            similar_dups.add(path2)
                            print(f"Foto augmentasi dengan nilai ({score}): {path2.name} == {path1.name}")

        print(f"Check Pertama: {total_checked - cross_checked} comparisons")
        print(f"Cross-Check: {cross_checked} comparisons")

    print(f"\nSelasai dengan pengecekan sebanyak {total_checked} untuk jumlah image {len(hash_data)}")
    print("Foto yang ternyata duplikat: ", len(exact_dups))
    print("Foto yang merupakan augmentasi: ", len(similar_dups))
    print("Total dirty data: ", len(exact_dups) + len(similar_dups))

    all_dups = set()
    all_dups.update(exact_dups)
    all_dups.update(similar_dups)

    if len(all_dups) == 0:
        print("Tidak ada duplikat HOREEE!!!")
        return None
    
    if preview_only:
        print("MODE PREVIEW - Tidak ada file yang dihapus")
        print("jalankan dengan preview_only=False untuk menghapus file")
        return None
    
    dups_path = base_path/"_dup_files"
    dups_path.mkdir(exist_ok=True)

    moved_count = 0
    for path in all_dups:
        try:
            dest = dups_path/path.name
            if dest.exists():
                dest = dups_path/f"{path.stem}_{path.parent.name}{path.suffix}"

            shutil.move(str(path), str(dest))
            moved_count += 1

            label_path = path.parent.parent / "labels" / f"{path.stem}.txt"
            if label_path.exists():
                label_dest = dups_path / f"{path.stem}.txt"
                if label_dest.exists():
                    label_dest = dups_path / f"{path.stem}_{path.parent.name}.txt"
                shutil.move(str(label_path), str(label_dest))

        except Exception as err:
            print("Error ketika memindahkan file duplikat: ", err)
            return None
        
    print("\nSELESAI!")
    print(f"{moved_count} file dipindahkan ke: {dups_path}")
    print(f"Dataset awal: {total_images} gambar")
    print(f"Dataset akhir: {total_images - moved_count} gambar")
    print(f"Pengurangan: {moved_count/total_images*100:.1f}%")

def process_images(path: Path):
    try:
        with Image.open(path) as img:
            ahash = imagehash.average_hash(img, hash_size=8)
            dhash = imagehash.dhash(img, hash_size=8)
            phash = imagehash.phash(img, hash_size=8)

            file_size = path.stat().st_size

            return (ahash, dhash, phash, file_size, path)

    except Exception as err:
        print("Error ketika proses image: ", err)
        return None

def cal_similarity_score(hash1: Tuple|List, hash2: Tuple|List):
    ah1, dh1, ph1, _, _ = hash1
    ah2, dh2, ph2, _, _ = hash2

    score = ((ah1 - ah2) * 2 + (dh1 - dh2) * 1 + (ph1 - ph2) * 1)

    return score

