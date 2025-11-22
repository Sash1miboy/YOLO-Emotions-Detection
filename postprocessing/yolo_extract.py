import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import yaml
import csv
import torch

class YOLOMetricsExtractor:
    """Extract metrics dari YOLO models (Strict Version: Stops on Error)"""
    
    def __init__(self, base_dir: str, data_yaml: str):
        self.base_dir = Path(base_dir)
        self.data_yaml = data_yaml
        self.results = []
        
    def find_all_models(self):
        print(f"Scanning folder: {self.base_dir}")
        models = []
        
        for comparison_folder in sorted(self.base_dir.glob("emotion_detection_yolov*_comparison")):
            print(f"comparison folder: {comparison_folder.name}")
            for train_folder in sorted(comparison_folder.glob("yolov*_train_results")):
                weight_file = train_folder / "weights" / "best.pt"
                if weight_file.exists():
                    model_info = self._parse_model_path(train_folder, weight_file)
                    if model_info:
                        models.append(model_info)
                        print(f"ketemu model: {model_info['name']}")
                    else:
                        print(f"model gak ketemu di folder: {train_folder.name}")
        return sorted(models, key=lambda x: (x['version'], x['size_order']))
    
    def _parse_model_path(self, train_folder: Path, weight_path: Path):
        folder_name = train_folder.name.lower()
        version = None
        for v in ['8', '9', '10', '11', '12']:
            if f'yolo{v}' in folder_name or f'yolov{v}' in folder_name:
                version = v
                break
        if not version: return None
        
        size = None
        size_order_map = {'n': 0, 's': 1, 'm': 2, 'l': 3, 'x': 4, 't': 5}
        for s in ['n', 's', 'm', 'l', 'x','t']:
            if f'yolo{version}{s}' in folder_name or f'yolov{version}{s}' in folder_name:
                size = s
                break
        if not size: return None
        
        return {
            'version': version,
            'size': size,
            'size_order': size_order_map[size],
            'name': f'YOLOv{version}{size}',
            'folder': train_folder,
            'path': weight_path
        }
        
    def extract_accuracy_metrics(self, model, model_name: str, results_csv: Path):
        """Extract accuracy metrics. Error = Stop."""
        print(f"extract metrics dari {model_name}...")
        
        train_metrics = None
        
        # Coba baca CSV (Fallback, kalau error cuma print warning, TIDAK STOP)
        if results_csv and results_csv.exists():
            try:
                print(f"Reading from results.csv...")
                res = pd.read_csv(results_csv)
                col = "metrics/mAP50-95(B)"
                
                # Handling spasi nama kolom (Optional tapi recommended)
                res.columns = res.columns.str.strip()
            
                if col in res.columns:
                    res_sort = res.sort_values(col, ascending=False)
                    best_row = res_sort.iloc[0]
                    
                    metrics = {
                        'Precision': float(best_row.get('metrics/precision(B)', 0)),
                        'Recall': float(best_row.get('metrics/recall(B)', 0)),
                        'mAP50': float(best_row.get('metrics/mAP50(B)', 0)),
                        'mAP50-95': float(best_row.get('metrics/mAP50-95(B)', 0)),
                    }
                    
                    if metrics['Precision'] + metrics['Recall'] > 0:
                        metrics['F1-Score'] = 2 * (metrics['Precision'] * metrics['Recall']) / (metrics['Precision'] + metrics['Recall'])
                    else:
                        metrics['F1-Score'] = 0.0
                    
                    print(f"dapat dari epoch: {int(best_row['epoch'])}")
                    train_metrics = metrics
            except Exception as err:
                print(f"⚠️ Warning: Gagal baca results.csv ({err}), lanjut ke validasi manual...")
        
        # --- VALIDASI MANUAL (NO TRY-EXCEPT -> Error = STOP) ---
        print(f"running best.pt dari {model_name}")
        
        # PARAMETER ASLI KAMU (TIDAK DIUBAH)
        test_params = {
            "imgsz": 640,
            "batch": 32,
            "save_json": True,
            "device": 0,
            "plots": True,
            "split": 'test',
            "verbose": True,
            "save_txt": True,
            "save_conf": True,
        }
        
        # Execute (Akan crash kalau error)
        results = model.val(data=self.data_yaml, **test_params)
        
        box = results.box
        metrics = {
            'Precision': float(box.mp), # Ubah ke box.mp jika box.p error (di ultralytics terbaru biasanya box.mp)
            'Recall': float(box.mr),    # Ubah ke box.mr jika error
            'mAP50': float(box.map50),
            'mAP50-95': float(box.map),
        }
        
        # Note: Ultralytics terbaru kadang pakai .mp dan .mr untuk mean precision/recall
        # Jika box.p error, ganti jadi box.mp di atas.
        
        if metrics['Precision'] + metrics['Recall'] > 0:
            metrics['F1-Score'] = 2 * (metrics['Precision'] * metrics['Recall']) / (metrics['Precision'] + metrics['Recall'])
        else:
            metrics['F1-Score'] = 0.0
        
        print(f"Selesai extract accuracy metrics")
        return metrics, train_metrics

    def extract_efficiency_metrics(self, model, model_name: str, num_samples: int = 100):
        """Extract efficiency metrics. Error = Stop."""
        print(f"Benchmarking efficiency buat {model_name}...")
        
        with open(self.data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
        
        if 'path' in data_config:
            dataset_path = Path(data_config['path'])
        else:
            dataset_path = Path(self.data_yaml).parent
        
        test_path = data_config.get('test', 'test/images')
        if not Path(test_path).is_absolute():
            test_images_dir = dataset_path / test_path
        else:
            test_images_dir = Path(test_path)
        
        test_images = list(test_images_dir.glob('*.jpg'))
        if not test_images: test_images = list(test_images_dir.glob('*.png'))
        if not test_images: test_images = list(test_images_dir.glob('*.jpeg'))
        
        if len(test_images) == 0:
            raise FileNotFoundError(f"Gak ada image di {test_images_dir}")
        
        test_images = test_images[:num_samples]
        
        print(f"🔥 Warming up GPU...")
        for _ in range(10):
            _ = model.predict(test_images[0], verbose=False)
        
        print(f"Benchmarking dengan {len(test_images)} images...")
        
        preprocess_times = []
        inference_times = []
        postprocess_times = []
        
        for img_path in tqdm(test_images, desc="Benchmarking", leave=False):
            # PARAMETER ASLI KAMU (TIDAK DIUBAH)
            predict_params = {
                "imgsz": 640,
                "device": 0,
                "batch": 1,
                "verbose": True,
                "stream": False
            }
            
            # Execute (Akan crash kalau error)
            results = model.predict(img_path, **predict_params)
            
            if hasattr(results[0], 'speed') and results[0].speed:
                speed = results[0].speed
                preprocess_times.append(speed['preprocess'])
                inference_times.append(speed['inference'])
                postprocess_times.append(speed['postprocess'])
        
        if len(preprocess_times) > 0:
            metrics = {
                'Preprocessing (ms)': float(np.mean(preprocess_times)),
                'Inference (ms)': float(np.mean(inference_times)),
                'Postprocessing (ms)': float(np.mean(postprocess_times)),
            }
            print(f"Benchmark completed")
            return metrics
        else:
            raise RuntimeError("No timing data collected")

    def extract_model_info(self, model):
        """
        Extract model info (STRICT MODE).
        Fix: Index [1] adalah Params, Index [3] adalah GFLOPs.
        Jika info kosong atau format salah -> LANGSUNG ERROR/STOP.
        """
        # Ambil info (verbose=True kadang diperlukan untuk YOLOv10 biar gak None)
        info = model.info(verbose=True) 
        
        # Cek 1: Apakah info None? Jika ya, RAISE ERROR (Berhenti)
        if info is None:
            raise ValueError(f"❌ FATAL: Model mengembalikan info() kosong (None). Tidak bisa lanjut.")
            
        # Cek 2: Apakah format list/tuple?
        if not isinstance(info, (tuple, list)):
             raise TypeError(f"❌ FATAL: Format info bukan list/tuple, tapi {type(info)}. Tidak bisa lanjut.")

        # Cek 3: Apakah panjang data cukup? (Butuh minimal 4 item: layers, params, grad, flops)
        if len(info) < 4:
             raise ValueError(f"❌ FATAL: Data info kurang lengkap (len={len(info)}). Harapan minimal 4 item. Isi: {info}")

        # Jika semua lolos, ambil data yang BENAR (Tanpa try-except)
        # Index 1 = Parameters
        # Index 3 = GFLOPs
        return {
            'Parameters (M)': info[1] / 1e6, 
            'GFLOPs': info[3]
        }
    
    def process_single_model(self, model_info: dict, benchmark_samples: int = 100):
        """Process single model. Error = Stop."""
        model_name = model_info['name']
        print(f"\n{'='*70}\n🔍 Processing: {model_name}\n{'='*70}")
        
        print(f"  📦 Loading model...")
        model = YOLO(model_info['path'])
        
        result = {
            'Model': model_name,
            'Version': model_info['version'],
            'Size': model_info['size'],
        }
        
        # 1. Model info
        result.update(self.extract_model_info(model))
        
        # 2. Accuracy metrics
        results_csv = model_info['folder'] / "results.csv"
        accuracy_metrics, train_metrics = self.extract_accuracy_metrics(
            model, model_name, results_csv
        )
        result.update(accuracy_metrics)
        
        # 3. Efficiency metrics
        efficiency_metrics = self.extract_efficiency_metrics(
            model, model_name, num_samples=benchmark_samples
        )
        result.update(efficiency_metrics)
        
        # 4. Derived metrics
        result['Total Time (ms)'] = (
            result['Preprocessing (ms)'] + 
            result['Inference (ms)'] + 
            result['Postprocessing (ms)']
        )
        
        result['FPS'] = 1000 / result['Total Time (ms)'] if result['Total Time (ms)'] > 0 else 0.0
        
        if result['GFLOPs'] > 0 and result['FPS'] > 0:
            result['Efficiency Score'] = (result['mAP50-95'] * result['FPS']) / result['GFLOPs']
        else:
            result['Efficiency Score'] = 0.0
        
        print(f"\n  ✅ {model_name} completed successfully!")
        print(f"     mAP50-95:  {result['mAP50-95']:.4f}")
        print(f"     FPS:       {result['FPS']:.2f}")
        
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        
        return result
    
    def process_all_models(self, benchmark_samples: int = 100):
        print("\n" + "="*70 + "\n  YOLO MODELS METRICS (STRICT MODE)\n" + "="*70)
        
        models = self.find_all_models()
        if not models: print("❌ No models found!"); return None
        
        print(f"\n✅ Found {len(models)} models")
        
        results = []
        for i, model_info in enumerate(models, 1):
            print(f"\n[{i}/{len(models)}]")
            # Error will crash here
            result = self.process_single_model(model_info, benchmark_samples)
            results.append(result)
            
        if not results: return None
        
        df = pd.DataFrame(results)
        df = df.sort_values(['Version', 'Size'])
        self.results_df = df
        return df
    
    def save_results(self, output_dir: Path):
        if output_dir is None: output_dir = self.base_dir / "metrics_results"
        else: output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        csv_path = output_dir / "yolo_metrics_complete.csv"
        self.results_df.to_csv(csv_path, index=False, float_format='%.4f')
        print(f"\n💾 Results saved to: {csv_path}")
        
        try:
            excel_path = output_dir / "yolo_metrics_complete.xlsx"
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                self.results_df.to_excel(writer, index=False, sheet_name='All Metrics', float_format='%.4f')
            print(f"💾 Excel saved to: {excel_path}")
        except Exception as e:
            print(f"⚠️  Could not save Excel: {e}")

    def display_summary(self):
        if hasattr(self, 'results_df') and self.results_df is not None:
            print("\nSummary:")
            print(self.results_df[['Model', 'mAP50-95', 'FPS', 'Efficiency Score']].to_string(index=False))