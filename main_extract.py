import os
from pathlib import Path

from postprocessing.new_yolo_extract import YOLOClassMetricsExtractor

if __name__ == "__main__":
    BASE_DIR = r"C:\Users\User\Desktop\YOLO-Results-Fixed-AffectNet"
    DATA_YAML = r"E:\thesis-model\Dataset\AffectNet-YOLO-Format-Preprocessed\data.yaml"

    if not os.path.exists(BASE_DIR):
        print(f"Error: Folder {BASE_DIR} tidak ditemukan!")
        exit()

    if not os.path.exists(DATA_YAML):
        print(f"Error: File {DATA_YAML} tidak ditemukan!")
        exit(0)

    extractor_class = YOLOClassMetricsExtractor(BASE_DIR, DATA_YAML)

    df_results_class = extractor_class.process_all_models(benchmark_samples=891)

    if df_results_class is not None:
        extractor_class.display_summary()
        extractor_class.save_results(
            output_dir=Path(r"C:\Users\User\Desktop\YOLO-Emotions-Detection\metrics_res")
        )

        print("\nSelesai! Cek folder 'metrics_results' untuk file Excel/CSV.")
