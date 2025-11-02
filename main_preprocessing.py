from pathlib import Path
from preprocessing.analyze_dataset import analyze_dataset
from preprocessing.remove_duplicates import remove_duplicates
from preprocessing.stratified_split import stratified_split

path = Path(r"'data.yaml' path dari dataset")
out_path = Path(r"lokasi yang anda inginkan untuk dataset yang telah di preprocessing")

splits = (0.7, 0.2, 0.1)

print("\nCheck Dataset Before Preprocessing!\n")
analyze_dataset(str(path))

print("\nProses Penghapusan Duplikat/Foto yang sangat mirip\n")

remove_duplicates(
    yaml_path=str(path), 
    max_workers=8, 
    similiarity_threshold=15,
    exact_threshold=3, 
    preview_only=True, # ubah ke false kalau ingin hapus data (awas lama bisa 10 menitan)
    use_full_scan=False,
) 

print("\nProses Stratified Spltiing!\n")

stratified_split(str(path), str(out_path), splits, 8)

new_path = out_path / 'data.yaml'

print("\nCheck Dataset After Preprocessing!\n")
analyze_dataset(str(new_path))