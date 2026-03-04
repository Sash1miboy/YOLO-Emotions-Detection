import os
import cv2
import shutil
import yaml
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

source_dataset = r"E:\thesis-model\Dataset\AffectNet-YOLO-Format-Cleaned"
output_dataset = r"E:\thesis-model\Dataset\AffectNet-MP-BBox"
model_path = r"C:\Users\User\Desktop\YOLO-Emotions-Detection\preprocessing\mediapipe\blaze_face_short_range.tflite"


base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)


def process_affectnet():
    failed_images = []

    for split in ["train", "valid", "test"]:
        print(f"\nProcessing {split}...")

        src_img_dir = os.path.join(source_dataset, split, "images")
        src_lbl_dir = os.path.join(source_dataset, split, "labels")

        dst_img_dir = os.path.join(output_dataset, split, "images")
        dst_lbl_dir = os.path.join(output_dataset, split, "labels")

        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lbl_dir, exist_ok=True)

        for img_name in os.listdir(src_img_dir):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            src_img_path = os.path.join(src_img_dir, img_name)
            src_lbl_path = os.path.join(src_lbl_dir, os.path.splitext(img_name)[0] + ".txt")

            if not os.path.exists(src_lbl_path):
                continue

            image = cv2.imread(src_img_path)
            if image is None:
                continue

            img_h, img_w, _ = image.shape

            # Ambil class lama
            with open(src_lbl_path, "r") as f:
                old_line = f.readline().strip()

            if not old_line:
                continue

            class_id = int(old_line.split()[0])

            # Detect face
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            detection_result = detector.detect(mp_image)

            if detection_result.detections:
                bbox = detection_result.detections[0].bounding_box

                w_norm = bbox.width / img_w
                h_norm = bbox.height / img_h
                x_center = (bbox.origin_x + bbox.width / 2) / img_w
                y_center = (bbox.origin_y + bbox.height / 2) / img_h

                # Clamp
                w_norm = min(max(w_norm, 0), 1)
                h_norm = min(max(h_norm, 0), 1)
                x_center = min(max(x_center, 0), 1)
                y_center = min(max(y_center, 0), 1)

                # Copy image
                shutil.copy2(src_img_path, os.path.join(dst_img_dir, img_name))

                # Save new label
                dst_lbl_path = os.path.join(dst_lbl_dir, os.path.splitext(img_name)[0] + ".txt")
                with open(dst_lbl_path, "w") as f:
                    f.write(
                        f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
                    )
            else:
                failed_images.append(src_img_path)

    print("\nSelesai generate dataset baru!")
    print(f"Gagal detect: {len(failed_images)}")

    if failed_images:
        with open("failed_affectnet_detection.txt", "w") as f:
            for item in failed_images:
                f.write(item + "\n")
        print("List gagal disimpan.")

    create_yaml()


def create_yaml():
    data_config = {
        "path": output_dataset,
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 8,
        "names": [
            "Anger",
            "Contempt",
            "Disgust",
            "Fear",
            "Happy",
            "Neutral",
            "Sad",
            "Surprise",
        ],
    }

    yaml_path = os.path.join(output_dataset, "data.yaml")

    with open(yaml_path, "w") as f:
        yaml.dump(data_config, f, default_flow_style=True)

    print(f"data.yaml dibuat di {yaml_path}")


if __name__ == "__main__":
    process_affectnet()