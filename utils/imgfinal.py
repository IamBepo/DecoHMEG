import cv2
import numpy as np


def formula_image_cropper(input_path, output_path, padding=10, brightness_threshold=200):
    # 读取原始彩色图像
    raw_img = cv2.imread(input_path)
    if raw_img is None:
        raise FileNotFoundError()
    img_h, img_w = raw_img.shape[:2]

    gray_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)

    _, binary_mask = cv2.threshold(gray_img, brightness_threshold, 255, cv2.THRESH_BINARY)
    stroke_y, stroke_x = np.where(binary_mask == 0)

    if len(stroke_x) == 0:
        cv2.imwrite(output_path, raw_img)
        return

    x_min, x_max = stroke_x.min(), stroke_x.max()
    y_min, y_max = stroke_y.min(), stroke_y.max()

    crop_left = max(0, x_min - padding)
    crop_right = min(img_w, x_max + padding)
    crop_top = max(0, y_min - padding)
    crop_bottom = min(img_h, y_max + padding)

    final_img = raw_img[crop_top:crop_bottom, crop_left:crop_right]
    cv2.imwrite(output_path, final_img)
    print(f"裁剪完成：{input_path} → {output_path}")


if __name__ == "__main__":
    import os

    input_folder = r"..\bak"
    output_folder = r"..\final"
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            in_path = os.path.join(input_folder, filename)
            out_path = os.path.join(output_folder, filename)
            formula_image_cropper(in_path, out_path, padding=10)