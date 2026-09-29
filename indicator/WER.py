import os
import re
import csv
import logging

INPUT_LOG_PATH = r"C:\Users\11321\Desktop\LLMs\HG\Main\indicator\test_in_printer\seedream_2019_ocr_evaluation.log"
RESULT_CSV = "./seedream_2019_wer_recalculated_results.csv"
OUTPUT_LOG = "./seedream_2019_wer_recalculation.log"

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


def tokenize_latex(latex_str: str) -> list:
    clean = re.sub(r'\s+', '', latex_str)
    pattern = re.compile(r'\\[a-zA-Z]+|\\.|[a-zA-Z0-9]|[^\s]')
    return pattern.findall(clean)


def calculate_wer(reference_tokens: list, hypothesis_tokens: list) -> float:
    ref_len = len(reference_tokens)
    hyp_len = len(hypothesis_tokens)

    if ref_len == 0:
        return 0.0 if hyp_len == 0 else 1.0
    dp = [[0] * (hyp_len + 1) for _ in range(ref_len + 1)]
    for i in range(ref_len + 1):
        dp[i][0] = i
    for j in range(hyp_len + 1):
        dp[0][j] = j

    for i in range(1, ref_len + 1):
        for j in range(1, hyp_len + 1):
            cost = 0 if reference_tokens[i-1] == hypothesis_tokens[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # deletion
                dp[i][j-1] + 1,      # insertion
                dp[i-1][j-1] + cost  # substitution
            )

    return dp[-1][-1] / ref_len


def parse_log_file(log_path: str) -> list:
    samples = []
    current_name = None
    current_gt = None
    current_pred = None

    sample_re = re.compile(r'Processing sample:\s+(\S+)')
    gt_re = re.compile(r'\[GT\]\s+raw:\s+(.*)$')
    pred_re = re.compile(r'\[PRED\]\s+raw:\s+(.*)$')

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Match new sample
            sample_match = sample_re.search(line)
            if sample_match:
                if current_name and current_gt is not None and current_pred is not None:
                    samples.append((current_name, current_gt, current_pred))
                current_name = sample_match.group(1)
                current_gt = None
                current_pred = None
                continue

            # Match GT line
            gt_match = gt_re.search(line)
            if gt_match and current_name:
                current_gt = gt_match.group(1).strip()
                continue

            # Match PRED line
            pred_match = pred_re.search(line)
            if pred_match and current_name:
                current_pred = pred_match.group(1).strip()
                continue
    if current_name and current_gt is not None and current_pred is not None:
        samples.append((current_name, current_gt, current_pred))

    return samples


def main():
    logger = setup_logger(OUTPUT_LOG)
    logger.info("=" * 60)
    logger.info("WER Recalculation Program Started")
    logger.info(f"Input log file: {os.path.abspath(INPUT_LOG_PATH)}")
    logger.info(f"Result CSV path: {os.path.abspath(RESULT_CSV)}")
    logger.info(f"Output log path: {os.path.abspath(OUTPUT_LOG)}")
    logger.info("Tokenization mode: unified LaTeX syntax tokenizer")
    logger.info("=" * 60)

    if not os.path.isfile(INPUT_LOG_PATH):
        logger.error(f"Input log file does not exist: {INPUT_LOG_PATH}")
        return

    logger.info("Parsing log file and extracting samples...")
    samples = parse_log_file(INPUT_LOG_PATH)
    if not samples:
        logger.error("No valid GT-PRED pairs found in log file")
        return
    logger.info(f"Successfully extracted {len(samples)} valid samples")

    logger.info("Recalculating WER for all samples...")
    result_rows = []
    wer_scores = []
    exact_count = 0

    for idx, (name, gt_raw, pred_raw) in enumerate(samples, 1):
        # Tokenize both sides with unified rules
        gt_tokens = tokenize_latex(gt_raw)
        pred_tokens = tokenize_latex(pred_raw)

        # Calculate WER
        wer = calculate_wer(gt_tokens, pred_tokens)
        is_exact = gt_tokens == pred_tokens

        if is_exact:
            exact_count += 1
        wer_scores.append(wer)

        result_rows.append([
            name,
            gt_raw,
            pred_raw,
            len(gt_tokens),
            len(pred_tokens),
            round(wer, 4),
            "是" if is_exact else "否"
        ])

        if idx % 100 == 0 or idx == len(samples):
            logger.info(f"  Processed {idx}/{len(samples)} samples")

    avg_wer = sum(wer_scores) / len(wer_scores) if wer_scores else 0.0
    exp_rate = exact_count / len(samples) if samples else 0.0

    logger.info("-" * 60)
    logger.info(f"Total valid samples: {len(samples)}")
    logger.info(f"Average WER: {avg_wer:.4f}")
    logger.info(f"Exact match count: {exact_count}")
    logger.info(f"Exact match rate (ExpRate): {exp_rate:.4f}")

    logger.info("Saving recalculated results to CSV...")
    try:
        with open(RESULT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['文件名', '真实LaTeX(原始)', '识别LaTeX(原始)', '真值token数', '识别token数', 'WER', '完全匹配'])
            writer.writerows(result_rows)
            writer.writerow([])
            writer.writerow(['统计项', '数值', '备注'])
            writer.writerow(['有效样本数', len(samples), ''])
            writer.writerow(['平均WER', round(avg_wer, 4), '越低越好，0表示完全一致'])
            writer.writerow(['完全匹配数', exact_count, ''])
            writer.writerow(['完全匹配率(ExpRate)', round(exp_rate, 4), '越高越好'])

        logger.info(f"Result CSV saved to: {os.path.abspath(RESULT_CSV)}")
    except Exception as e:
        logger.error(f"Failed to save CSV file: {str(e)}")

    logger.info("=" * 60)
    logger.info("WER recalculation completed")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
