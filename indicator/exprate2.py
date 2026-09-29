import os
import csv
import logging
import base64
import time
import re
from openai import OpenAI

# ==================== Model Configuration ====================
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen3.6-plus"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ==================== Path Configuration ====================
IMAGE_FOLDER =
CAPTION_FILE =
RESULT_CSV =
LOG_FILE =

REQUEST_INTERVAL = 1
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
CASE_INSENSITIVE_MATCH = True


# ==========================================================


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


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        base64_str = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(image_path)[1].lower().replace(".", "")
    ext = "jpeg" if ext == "jpg" else ext
    return f"data:image/{ext};base64,{base64_str}"


def call_qwen_vl(image_path: str) -> str:
    try:
        base64_image = encode_image_to_base64(image_path)
        prompt = "Recognize the mathematical formula in the image. Output ONLY the corresponding LaTeX code. No explanation, no markdown markers, no extra text, no surrounding dollar signs."

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": base64_image}}
                    ]
                }
            ],
            temperature=0
        )
        result = response.choices[0].message.content.strip()
        result = re.sub(r"```latex|```", "", result, flags=re.IGNORECASE).strip()
        result = re.sub(r"\s+", " ", result).strip()
        return result
    except Exception as e:
        logging.error(f"Model call failed: {str(e)}")
        return ""


def load_caption_mapping(caption_path: str) -> dict:
    mapping = {}
    if not os.path.exists(caption_path):
        logging.error(f"Caption file not found: {caption_path}")
        return mapping

    with open(caption_path, "r", encoding="utf-8-sig") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            if "\t" in line:
                parts = line.split("\t", 1)
            else:
                parts = line.split(maxsplit=1)

            if len(parts) != 2:
                logging.warning(f"Line {line_num} format invalid, skipped: {line[:50]}")
                continue

            filename, latex = parts
            filename = filename.strip()
            stem = os.path.splitext(filename)[0]

            if CASE_INSENSITIVE_MATCH:
                stem = stem.lower()

            mapping[stem] = latex.strip()

    if mapping:
        sample_keys = list(mapping.keys())[:5]
        logging.info(f"First 5 caption keys: {', '.join(sample_keys)}")
    return mapping


def normalize_latex_for_compare(latex_str: str) -> str:
    normalized = re.sub(r"\s+", "", latex_str)
    return normalized


def calculate_cer(reference: str, hypothesis: str, debug: bool = False) -> float:
    ref_norm = normalize_latex_for_compare(reference)
    hyp_norm = normalize_latex_for_compare(hypothesis)

    ref_chars = list(ref_norm)
    hyp_chars = list(hyp_norm)

    if debug:
        logging.info(f"  [GT]  raw: {reference}")
        logging.info(f"  [PRED] raw: {hypothesis}")
        logging.info(f"  [GT]  normalized: {ref_norm} (len={len(ref_chars)})")
        logging.info(f"  [PRED] normalized: {hyp_norm} (len={len(hyp_chars)})")

    rows = len(ref_chars) + 1
    cols = len(hyp_chars) + 1
    dp = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        dp[i][0] = i
    for j in range(cols):
        dp[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if ref_chars[i - 1] == hyp_chars[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )

    edit_dist = dp[-1][-1]

    if debug:
        logging.info(f"  Edit distance: {edit_dist} | Reference length: {len(ref_chars)}")

    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0
    return edit_dist / len(ref_chars)


def main():
    logger = setup_logger(LOG_FILE)
    logger.info("=" * 60)
    logger.info("LaTeX OCR Evaluation Program Started")
    logger.info(f"Image folder: {os.path.abspath(IMAGE_FOLDER)}")
    logger.info(f"Caption file: {os.path.abspath(CAPTION_FILE)}")
    logger.info(f"Result CSV: {os.path.abspath(RESULT_CSV)}")
    logger.info(f"Log file: {os.path.abspath(LOG_FILE)}")
    logger.info(f"Evaluation model: {MODEL_NAME}")
    logger.info(f"Metric: CER (Character Error Rate)")
    logger.info("=" * 60)

    # Path validation
    if not os.path.isdir(IMAGE_FOLDER):
        logger.error(f"Image folder does not exist: {IMAGE_FOLDER}")
        return
    if not os.path.isfile(CAPTION_FILE):
        logger.error(f"Caption file does not exist: {CAPTION_FILE}")
        return

    # Load ground truth
    logger.info("Loading caption ground truth...")
    caption_map = load_caption_mapping(CAPTION_FILE)
    if not caption_map:
        logger.error("No valid caption entries loaded, please check file format")
        return
    logger.info(f"Loaded {len(caption_map)} ground truth entries")

    # Scan image files
    image_files = []
    for f in os.listdir(IMAGE_FOLDER):
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
            image_files.append(f)

    if not image_files:
        logger.error("No image files found in target folder")
        return
    logger.info(f"Found {len(image_files)} image files in folder")

    sample_stems = [os.path.splitext(f)[0].lower() if CASE_INSENSITIVE_MATCH else os.path.splitext(f)[0] for f in
                    image_files[:5]]
    logger.info(f"First 5 image stems: {', '.join(sample_stems)}")

    # Match images with captions
    matched_pairs = []
    unmatched_images = []
    for img_file in image_files:
        stem = os.path.splitext(img_file)[0]
        if CASE_INSENSITIVE_MATCH:
            stem = stem.lower()

        if stem in caption_map:
            matched_pairs.append((stem, os.path.join(IMAGE_FOLDER, img_file), caption_map[stem]))
        else:
            unmatched_images.append(img_file)

    if unmatched_images:
        logger.warning(
            f"Images without matching caption ({len(unmatched_images)}): {', '.join(unmatched_images[:10])} ...")
    if not matched_pairs:
        logger.error("No valid image-caption pairs, program exiting")
        logger.info("Hint: Check if filename stems in caption match image file names")
        return

    logger.info(f"Successfully matched {len(matched_pairs)} image-caption pairs")
    logger.info("-" * 60)

    # Batch evaluation
    results = []
    cer_list = []
    exact_count = 0
    success = 0
    failed = 0

    for idx, (name, img_path, gt_latex) in enumerate(matched_pairs, 1):
        logger.info(f"[{idx}/{len(matched_pairs)}] Processing sample: {name}")
        try:
            # Model inference
            pred_latex = call_qwen_vl(img_path)
            if not pred_latex:
                failed += 1
                results.append([name, gt_latex, "", "识别失败", ""])
                logger.error("  Recognition failed: empty response")
                time.sleep(REQUEST_INTERVAL)
                continue

            # Calculate CER with debug print
            cer = calculate_cer(gt_latex, pred_latex, debug=True)
            is_exact = normalize_latex_for_compare(gt_latex) == normalize_latex_for_compare(pred_latex)

            if is_exact:
                exact_count += 1
            cer_list.append(cer)
            success += 1

            results.append([name, gt_latex, pred_latex, round(cer, 4), "是" if is_exact else "否"])
            logger.info(f"  Success | CER: {cer:.4f} | Exact match: {is_exact}")

        except Exception as e:
            failed += 1
            results.append([name, gt_latex, "", "处理异常", str(e)])
            logger.error(f"  Processing failed: {str(e)}")

        time.sleep(REQUEST_INTERVAL)

    # Compute statistics
    logger.info("-" * 60)
    avg_cer = sum(cer_list) / len(cer_list) if cer_list else 0.0
    exp_rate = exact_count / success if success > 0 else 0.0

    # Save CSV result
    logger.info("Saving evaluation results to CSV...")
    try:
        with open(RESULT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['文件名', '真实LaTeX', '识别LaTeX', 'CER', '完全匹配'])
            writer.writerows(results)

            # Statistics section
            writer.writerow([])
            writer.writerow(['统计项', '数值'])
            writer.writerow(['总样本数', len(matched_pairs)])
            writer.writerow(['识别成功数', success])
            writer.writerow(['识别失败数', failed])
            writer.writerow(['平均CER', round(avg_cer, 4)])
            writer.writerow(['完全匹配率(ExpRate)', round(exp_rate, 4)])

        logger.info(f"Result CSV saved to: {os.path.abspath(RESULT_CSV)}")
    except Exception as e:
        logger.error(f"Failed to save CSV file: {str(e)}")

    # Final summary
    logger.info("=" * 60)
    logger.info("Evaluation task completed")
    logger.info(f"Total samples: {len(matched_pairs)} | Succeeded: {success} | Failed: {failed}")
    if cer_list:
        logger.info(f"Average CER: {avg_cer:.4f}")
        logger.info(f"Exact Match Rate (ExpRate): {exp_rate:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
