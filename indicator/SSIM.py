import os
import csv
import logging
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# -------------------------- 配置参数 --------------------------
FOLDER_A = r"C:\Users\11321\Desktop\LLMs\HG\Main\model\GPT\2019"
FOLDER_B = r"C:\Users\11321\Desktop\LLMs\HV\data\2019\img"
RESULT_CSV = "./gpt_2019_ssim_results.csv"
LOG_FILE = "./gpt_2019_ssim_calculate.log"

# 图像归一化参数
TARGET_HEIGHT = 128
CANVAS_SIZE = (512, 256)
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

def setup_logger(log_path: str):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def normalize_img(img_path: str, target_h: int = 128, canvas: tuple = (512, 256)) -> np.ndarray:
    img = Image.open(img_path).convert("L")
    arr = np.array(img)
    mask = arr < 250
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.full(canvas[::-1], 255, dtype=np.uint8)

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    crop = img.crop((x0, y0, x1, y1))

    w, h = crop.size
    new_w = max(1, int(w * target_h / h))
    crop = crop.resize((new_w, target_h), Image.Resampling.LANCZOS)

    cw, ch = canvas
    bg = Image.new("L", canvas, 255)
    paste_x = max(0, (cw - new_w) // 2)
    bg.paste(crop, (paste_x, (ch - target_h) // 2))

    return np.array(bg)


def calculate_ssim(img_path1: str, img_path2: str) -> float:
    img1_norm = normalize_img(img_path1, TARGET_HEIGHT, CANVAS_SIZE)
    img2_norm = normalize_img(img_path2, TARGET_HEIGHT, CANVAS_SIZE)
    return ssim(img1_norm, img2_norm, data_range=255)


def get_matched_pairs(folder1: str, folder2: str):
    files1 = {}
    for f in os.listdir(folder1):
        ext = os.path.splitext(f)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            stem = os.path.splitext(f)[0]
            files1[stem] = (f, os.path.join(folder1, f))

    files2 = {}
    for f in os.listdir(folder2):
        ext = os.path.splitext(f)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            stem = os.path.splitext(f)[0]
            files2[stem] = (f, os.path.join(folder2, f))

    common_stems = sorted(set(files1.keys()) & set(files2.keys()))
    pairs = [(stem, files1[stem][1], files2[stem][1]) for stem in common_stems]

    only_folder1 = sorted([files1[s][0] for s in set(files1.keys()) - set(files2.keys())])
    only_folder2 = sorted([files2[s][0] for s in set(files2.keys()) - set(files1.keys())])

    return pairs, only_folder1, only_folder2


def main():
    logger = setup_logger(LOG_FILE)
    logger.info("=" * 60)
    logger.info("SSIM Batch Calculation Program Started")
    logger.info(f"Folder A path: {os.path.abspath(FOLDER_A)}")
    logger.info(f"Folder B path: {os.path.abspath(FOLDER_B)}")
    logger.info(f"Result CSV path: {os.path.abspath(RESULT_CSV)}")
    logger.info(f"Log file path: {os.path.abspath(LOG_FILE)}")
    logger.info("=" * 60)

    if not os.path.isdir(FOLDER_A):
        logger.error(f"Folder A does not exist, please check path: {FOLDER_A}")
        return
    if not os.path.isdir(FOLDER_B):
        logger.error(f"Folder B does not exist, please check path: {FOLDER_B}")
        return

    logger.info("Scanning image files in both folders...")
    pairs, only_a, only_b = get_matched_pairs(FOLDER_A, FOLDER_B)

    if only_a:
        logger.warning(f"Files only in Folder A ({len(only_a)}): {', '.join(only_a)}")
    if only_b:
        logger.warning(f"Files only in Folder B ({len(only_b)}): {', '.join(only_b)}")

    if not pairs:
        logger.error("No matching image pairs found by file name, program exiting")
        return

    logger.info(f"Successfully matched {len(pairs)} image pairs")
    logger.info("-" * 60)

    ssim_scores = []
    result_rows = []
    success = 0
    failed = 0

    for idx, (filename, path_a, path_b) in enumerate(pairs, 1):
        logger.info(f"[{idx}/{len(pairs)}] Processing file: {filename}")
        try:
            score = calculate_ssim(path_a, path_b)
            ssim_scores.append(score)
            result_rows.append([filename, round(score, 6)])
            success += 1
            logger.info(f"  Calculation succeeded, SSIM = {score:.6f}")
        except Exception as e:
            failed += 1
            result_rows.append([filename, "计算失败", str(e)])
            logger.error(f"  Calculation failed: {str(e)}")

    logger.info("-" * 60)
    logger.info("Saving result spreadsheet...")
    try:
        with open(RESULT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['文件名', 'SSIM值', '备注'])
            writer.writerows(result_rows)
            if ssim_scores:
                writer.writerow([])
                writer.writerow(['统计项', '数值'])
                writer.writerow(['成功样本数', success])
                writer.writerow(['失败样本数', failed])
                writer.writerow(['平均SSIM', round(np.mean(ssim_scores), 6)])
                writer.writerow(['最大SSIM', round(np.max(ssim_scores), 6)])
                writer.writerow(['最小SSIM', round(np.min(ssim_scores), 6)])
                writer.writerow(['标准差', round(np.std(ssim_scores), 6)])

        logger.info(f"Result CSV saved to: {os.path.abspath(RESULT_CSV)}")
    except Exception as e:
        logger.error(f"Failed to save CSV file: {str(e)}")
    logger.info("=" * 60)
    logger.info("All calculation tasks completed")
    logger.info(f"Total samples: {len(pairs)} | Succeeded: {success} | Failed: {failed}")
    if ssim_scores:
        logger.info(f"Average SSIM: {np.mean(ssim_scores):.6f}")
        logger.info(f"Max SSIM: {np.max(ssim_scores):.6f}")
        logger.info(f"Min SSIM: {np.min(ssim_scores):.6f}")
        logger.info(f"Standard Deviation: {np.std(ssim_scores):.6f}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
