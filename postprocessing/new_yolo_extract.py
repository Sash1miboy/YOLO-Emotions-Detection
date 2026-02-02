from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm
from ultralytics import YOLO


class YOLOClassMetricsExtractor:
    """Extract metrics YOLO: Accuracy, Efficiency Score, Training Time (HH:MM:SS) & Per-Class Data"""

    def __init__(self, base_dir: str, data_yaml: str):
        self.base_dir = Path(base_dir)
        self.data_yaml = data_yaml
        self.summary_results = []  # Row untuk Overall & Efficiency Score
        self.detailed_results = []  # Row untuk Per-Class Metrics

    def find_all_models(self):
        print(f"Scanning folder: {self.base_dir}")
        models = []

        for comparison_folder in sorted(
            self.base_dir.glob("emotion_detection_yolov*_comparison")
        ):
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
        return sorted(models, key=lambda x: (x["version"], x["size_order"]))

    def _parse_model_path(self, train_folder: Path, weight_path: Path):
        folder_name = train_folder.name.lower()

        version = None
        for v in ["8", "9", "10", "11", "12"]:
            if f"yolo{v}" in folder_name or f"yolov{v}" in folder_name:
                version = v
                break
        if not version:
            return None

        size = None
        size_order_map = {"n": 0, "s": 1, "m": 2, "l": 3, "x": 4, "t": 5, "c": 6}

        # Loop untuk mencari size model
        for s in ["n", "s", "m", "l", "x", "t", "c"]:
            if (
                f"yolo{version}{s}" in folder_name
                or f"yolov{version}{s}" in folder_name
            ):
                size = s
                break
        if not size:
            return None

        return {
            "version": version,
            "size": size,
            "size_order": size_order_map[size],
            "name": f"YOLOv{version}{size}",
            "folder": train_folder,
            "path": weight_path,
        }

    def extract_accuracy_metrics(self, model, model_name: str, results_csv: Path):
        """Extract accuracy metrics & Training Time from CSV. Error = Stop."""
        print(f"extract metrics dari {model_name}...")

        train_metrics = {}

        # --- 1. BACA CSV (Ambil Best Metric & Training Time) ---
        if results_csv and results_csv.exists():
            try:
                print("Reading from results.csv...")
                res = pd.read_csv(results_csv)
                res.columns = res.columns.str.strip()

                # A. Ambil Best Metrics (Berdasarkan mAP50-95 tertinggi)
                col = "metrics/mAP50-95(B)"
                if col in res.columns:
                    res_sort = res.sort_values(col, ascending=False)
                    best_row = res_sort.iloc[0]

                    metrics = {
                        "Precision": float(best_row.get("metrics/precision(B)", 0)),
                        "Recall": float(best_row.get("metrics/recall(B)", 0)),
                        "mAP50": float(best_row.get("metrics/mAP50(B)", 0)),
                        "mAP50-95": float(best_row.get("metrics/mAP50-95(B)", 0)),
                    }
                    p, r = metrics["Precision"], metrics["Recall"]
                    metrics["F1-Score"] = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
                    print(f"dapat dari epoch: {int(best_row['epoch'])}")
                    train_metrics = metrics

                # B. Ambil Training Time (Berdasarkan Epoch Terakhir) -> Convert to HH:MM:SS
                if "time" in res.columns and "epoch" in res.columns:
                    # Sort berdasarkan epoch (descending) untuk dapat yang terakhir
                    res_epoch_sort = res.sort_values("epoch", ascending=False)
                    last_row = res_epoch_sort.iloc[0]

                    total_seconds = float(last_row["time"])

                    # --- CONVERT KE HH:MM:SS ---
                    total_seconds = int(round(total_seconds))
                    hours = total_seconds // 3600
                    minutes = (total_seconds % 3600) // 60
                    seconds = total_seconds % 60

                    time_str = f"{hours:02}:{minutes:02}:{seconds:02}"

                    train_metrics["Training Time"] = time_str  # type: ignore
                    print(f"⏱️ Training Time (Last Epoch): {time_str}")

            except Exception as err:
                print(
                    f"⚠️ Warning: Gagal baca results.csv ({err}), lanjut ke validasi manual..."
                )

        # --- 2. VALIDASI MANUAL (Untuk Per-Class Metrics) ---
        print(f"running best.pt dari {model_name} untuk Per-Class Data...")

        test_params = {
            "imgsz": 640,
            "batch": 32,
            "save_json": True,
            "device": 0,
            "plots": False,
            "split": "test",
            "verbose": True,
            "save_txt": False,
            "save_conf": False,
        }

        results = model.val(data=self.data_yaml, **test_params)
        box = results.box

        # A. Overall Metrics
        val_metrics = {
            "Precision": float(box.mp),
            "Recall": float(box.mr),
            "mAP50": float(box.map50),
            "mAP50-95": float(box.map),
        }
        p, r = val_metrics["Precision"], val_metrics["Recall"]
        val_metrics["F1-Score"] = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

        # B. Per-Class Metrics
        per_class_list = []
        names = model.names

        try:
            ap50_95s = box.maps
            ap50s = box.ap50 if hasattr(box, "ap50") else box.all_ap[:, 0]
            ps = box.p
            rs = box.r
            indices = box.ap_class_index

            for i, class_idx in enumerate(indices):
                class_name = names[class_idx]

                # Safety check length
                p_val = ps[i] if len(ps) > i else 0.0
                r_val = rs[i] if len(rs) > i else 0.0
                map50_val = ap50s[i] if len(ap50s) > i else 0.0
                map95_val = ap50_95s[i] if len(ap50_95s) > i else 0.0

                # Hitung F1 Per Class
                f1_val = (
                    2 * (p_val * r_val) / (p_val + r_val)
                    if (p_val + r_val) > 0
                    else 0.0
                )

                per_class_list.append(
                    {
                        "Class": class_name,
                        "Precision": float(p_val),
                        "Recall": float(r_val),
                        "F1-Score": float(f1_val),
                        "mAP50": float(map50_val),
                        "mAP50-95": float(map95_val),
                    }
                )

        except Exception as e:
            print(f"⚠️ Gagal extract per-class data: {e}")

        print(f"Selesai extract metrics (Overall + {len(per_class_list)} Classes)")
        return val_metrics, train_metrics, per_class_list

    def extract_efficiency_metrics(
        self, model, model_name: str, num_samples: int = 100
    ):
        """Extract efficiency metrics. Error = Stop."""
        print(f"Benchmarking efficiency buat {model_name}...")

        with open(self.data_yaml, "r") as f:
            data_config = yaml.safe_load(f)

        dataset_path = Path(data_config.get("path", Path(self.data_yaml).parent))
        test_path = data_config.get("test", "test/images")

        if not Path(test_path).is_absolute():
            test_images_dir = dataset_path / test_path
        else:
            test_images_dir = Path(test_path)

        test_images = (
            list(test_images_dir.glob("*.jpg"))
            + list(test_images_dir.glob("*.png"))
            + list(test_images_dir.glob("*.jpeg"))
        )

        if len(test_images) == 0:
            raise FileNotFoundError(f"Gak ada image di {test_images_dir}")

        test_images = test_images[:num_samples]

        print("🔥 Warming up GPU...")
        for _ in range(10):
            _ = model.predict(test_images[0], verbose=False)

        print(f"Benchmarking dengan {len(test_images)} images...")
        preprocess_times, inference_times, postprocess_times = [], [], []

        predict_params = {
            "imgsz": 640,
            "device": 0,
            "batch": 1,
            "verbose": False,
            "stream": False,
        }

        for img_path in tqdm(test_images, desc="Benchmarking", leave=False):
            results = model.predict(img_path, **predict_params)
            if hasattr(results[0], "speed") and results[0].speed:
                speed = results[0].speed
                preprocess_times.append(speed["preprocess"])
                inference_times.append(speed["inference"])
                postprocess_times.append(speed["postprocess"])

        if not preprocess_times:
            raise RuntimeError("No timing data collected")

        return {
            "Preprocessing (ms)": float(np.mean(preprocess_times)),
            "Inference (ms)": float(np.mean(inference_times)),
            "Postprocessing (ms)": float(np.mean(postprocess_times)),
        }

    def extract_model_info(self, model):
        """Extract info (STRICT MODE)."""
        info = model.info(verbose=True)
        if info is None:
            raise ValueError("❌ FATAL: Info None")
        if not isinstance(info, (tuple, list)):
            raise TypeError("❌ FATAL: Info format wrong")
        if len(info) < 4:
            raise ValueError("❌ FATAL: Info length < 4")
        return {"Parameters (M)": info[1] / 1e6, "GFLOPs": info[3]}

    def process_single_model(self, model_info: dict, benchmark_samples: int = 100):
        """Process single model & Collect all data."""
        model_name = model_info["name"]
        print(f"\n{'=' * 70}\n🔍 Processing: {model_name}\n{'=' * 70}")

        print("  📦 Loading model...")
        model = YOLO(model_info["path"])

        # --- DATA OVERALL ---
        summary_row = {
            "Model": model_name,
            "Version": model_info["version"],
            "Size": model_info["size"],
        }
        summary_row.update(self.extract_model_info(model))

        # Extract Accuracy & Training Time
        results_csv = model_info["folder"] / "results.csv"
        val_metrics, train_metrics, per_class_list = self.extract_accuracy_metrics(
            model, model_name, results_csv
        )

        summary_row.update(val_metrics)

        # Masukkan Training Time ke Summary Row
        if train_metrics and "Training Time" in train_metrics:
            summary_row["Training Time"] = train_metrics["Training Time"]
        else:
            summary_row["Training Time"] = "00:00:00"

        # Extract Efficiency
        eff_metrics = self.extract_efficiency_metrics(
            model, model_name, num_samples=benchmark_samples
        )
        summary_row.update(eff_metrics)

        # Derived Metrics
        total_ms = (
            summary_row["Preprocessing (ms)"]
            + summary_row["Inference (ms)"]
            + summary_row["Postprocessing (ms)"]
        )
        summary_row["Total Time (ms)"] = total_ms
        summary_row["FPS"] = round(1000 / total_ms) if total_ms > 0 else 0.0

        if summary_row["GFLOPs"] > 0 and summary_row["FPS"] > 0:
            summary_row["Efficiency Score"] = (
                summary_row["mAP50-95"] * summary_row["FPS"]
            ) / summary_row["GFLOPs"]
        else:
            summary_row["Efficiency Score"] = 0.0

        # --- SIMPAN KE LIST UTAMA ---
        self.summary_results.append(summary_row)

        # --- DATA PER CLASS ---
        for item in per_class_list:
            detailed_row = {
                "Model": model_name,
                "Version": model_info["version"],
                "Size": model_info["size"],
                **item,
            }
            self.detailed_results.append(detailed_row)

        print(f"\n  ✅ {model_name} completed!")
        print(f"     mAP50-95: {summary_row['mAP50-95']:.4f}")
        print(f"     Train Time: {summary_row['Training Time']}")
        print(f"     FPS:      {summary_row['FPS']:.2f}")
        print(f"     Eff.Score:{summary_row['Efficiency Score']:.4f}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return summary_row

    def process_all_models(self, benchmark_samples: int = 100):
        print(
            "\n"
            + "=" * 70
            + "\n  YOLO METRICS (SUMMARY + EFFICIENCY SCORE)\n"
            + "=" * 70
        )

        models = self.find_all_models()
        if not models:
            print("❌ No models found!")
            return None

        print(f"\n✅ Found {len(models)} models")

        for i, model_info in enumerate(models, 1):
            print(f"\n[{i}/{len(models)}]")
            self.process_single_model(model_info, benchmark_samples)

        return pd.DataFrame(self.summary_results)

    def save_results(self, output_dir: Path):
        if output_dir is None:
            output_dir = self.base_dir / "metrics_results"
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # 1. Save Summary
        df_summary = pd.DataFrame(self.summary_results)

        # Atur urutan kolom agar Training Time muncul di depan
        first_cols = [
            "Model",
            "Version",
            "Size",
            "Parameters (M)",
            "GFLOPs",
            "Training Time",
            "mAP50-95",
            "FPS",
            "Efficiency Score",
        ]
        cols = first_cols + [c for c in df_summary.columns if c not in first_cols]
        df_summary = df_summary[cols]

        df_summary.to_csv(
            output_dir / "yolo_metrics_summary.csv", index=False, float_format="%.4f"
        )
        print(f"\n💾 Summary saved to: {output_dir / 'yolo_metrics_summary.csv'}")

        # 2. Save Detailed (Per Class)
        if self.detailed_results:
            df_detail = pd.DataFrame(self.detailed_results)
            df_detail.to_csv(
                output_dir / "yolo_metrics_detailed.csv",
                index=False,
                float_format="%.4f",
            )
            print(f"💾 Detailed saved to: {output_dir / 'yolo_metrics_detailed.csv'}")

        # 3. Save Excel
        try:
            excel_path = output_dir / "yolo_metrics_complete.xlsx"
            with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                df_summary.to_excel(
                    writer, index=False, sheet_name="Summary", float_format="%.4f"
                )
                if self.detailed_results:
                    df_detail.to_excel(
                        writer,
                        index=False,
                        sheet_name="Per-Class Details",
                        float_format="%.4f",
                    )  # type: ignore
            print(f"💾 Excel saved to: {excel_path}")
        except Exception as e:
            print(f"⚠️ Could not save Excel: {e}")

    def display_summary(self):
        if self.summary_results:
            df = pd.DataFrame(self.summary_results)
            # Tampilkan Efficiency Score & Training Time
            cols = ["Model", "mAP50-95", "Training Time", "FPS", "Efficiency Score"]
            print("\nSummary:")
            print(df[[c for c in cols if c in df.columns]].to_string(index=False))
