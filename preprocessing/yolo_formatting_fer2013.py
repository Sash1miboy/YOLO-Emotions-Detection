import os

import cv2
import mediapipe as mp
import yaml
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

dataset_path = r"E:\thesis-model\Dataset\FER-2013"
output_path = r"E:\thesis-model\Dataset\FER-2013-YOLO-Format"
model_path = r"C:\Users\User\Desktop\YOLO-Emotions-Detection\preprocessing\mediapipe\blaze_face_short_range.tflite"

emotion_map = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
    "surprise": 6,
}

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)


def process_dataset():
    for split in ["train", "test"]:
        input_split_path = os.path.join(dataset_path, split)
        if not os.path.exists(input_split_path):
            continue

        img_dest = os.path.join(output_path, split, "images")
        lbl_dest = os.path.join(output_path, split, "labels")
        os.makedirs(img_dest, exist_ok=True)
        os.makedirs(lbl_dest, exist_ok=True)

        for emotion_name, class_id in emotion_map.items():
            folder_path = os.path.join(input_split_path, emotion_name)
            if not os.path.exists(folder_path):
                continue

            print(f"Memproses {split}/{emotion_name}...")

            for filename in os.listdir(folder_path):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    file_path = os.path.join(folder_path, filename)

                    image_cv = cv2.imread(file_path)
                    if image_cv is None:
                        continue

                    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB, data=image_rgb
                    )

                    detection_result = detector.detect(mp_image)

                    # Default Bounding Box (YOLO Format)
                    x_center, y_center, w_norm, h_norm = 0.5, 0.5, 1.0, 1.0

                    if detection_result.detections:
                        bbox = detection_result.detections[0].bounding_box
                        img_h, img_w, _ = image_cv.shape

                        w_box = bbox.width / img_w
                        h_box = bbox.height / img_h
                        x_center = (bbox.origin_x + (bbox.width / 2)) / img_w
                        y_center = (bbox.origin_y + (bbox.height / 2)) / img_h

                        w_norm = min(w_box, 1.0)
                        h_norm = min(h_box, 1.0)

                    new_filename = f"{emotion_name}_{filename}"
                    cv2.imwrite(os.path.join(img_dest, new_filename), image_cv)

                    txt_name = os.path.splitext(new_filename)[0] + ".txt"
                    with open(os.path.join(lbl_dest, txt_name), "w") as f:
                        f.write(
                            f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
                        )

    print("Konversi Selesai!")


def visualize_yolo_annotations(
    image_dir,
    label_dir,
    save_vis_path,
    emotion_map,
):
    os.makedirs(save_vis_path, exist_ok=True)

    for img_name in os.listdir(image_dir):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(image_dir, img_name)
        label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + ".txt")

        img = cv2.imread(img_path)
        if img is None or not os.path.exists(label_path):
            continue

        h, w, _ = img.shape

        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            cls, x, y, bw, bh = map(float, line.strip().split())
            cls = int(cls)

            x1, y1, x2, y2 = yolo_to_bbox(w, h, x, y, bw, bh)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = emotion_map.get(cls, "unknown")

            cv2.putText(
                img,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        cv2.imwrite(os.path.join(save_vis_path, img_name), img)

    print(f"Visualisasi selesai! Cek folder: {save_vis_path}")


def create_yaml():
    data_config = {
        "path": output_path,
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(emotion_map),
        "names": list(emotion_map.keys()),
    }

    yaml_file_path = os.path.join(output_path, "data.yaml")

    with open(yaml_file_path, "w") as f:
        yaml.dump(data_config, f, default_flow_style=True)

    print(f"File {yaml_file_path} berhasil dibuat!")


def yolo_to_bbox(img_w, img_h, x, y, w, h):
    x1 = int((x - w / 2) * img_w)
    y1 = int((y - h / 2) * img_h)
    x2 = int((x + w / 2) * img_w)
    y2 = int((y + h / 2) * img_h)
    return x1, y1, x2, y2


if __name__ == "__main__":
    process_dataset()
    create_yaml()
