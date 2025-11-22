import os
from pathlib import Path
from postprocessing.yolo_extract import YOLOMetricsExtractor

if __name__ == "__main__":
    BASE_DIR = r"C:\Users\User\Desktop\YOLO-Results-Revised"
    DATA_YAML = r"C:\Users\User\Desktop\New-Human-Face-Detections\data.yaml"


    if not os.path.exists(BASE_DIR):
            print(f"Error: Folder {BASE_DIR} tidak ditemukan!")
            exit()
        
    if not os.path.exists(DATA_YAML):
        print(f"Error: File {DATA_YAML} tidak ditemukan!")
        exit(0)
        
    extractor = YOLOMetricsExtractor(BASE_DIR, DATA_YAML)

    df_results = extractor.process_all_models(benchmark_samples=891)

    if df_results is not None:
            # 4. Tampilkan Summary & Simpan
            extractor.display_summary()
            extractor.save_results(output_dir=Path(r"E:\thesis-model\YOLO-Emotions-Detection\metrics_res")) 
            
            print("\nSelesai! Cek folder 'metrics_results' untuk file Excel/CSV.")

