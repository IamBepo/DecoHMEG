import json
import os


def latex_to_flat_tokens(latex_str: str) -> list:
    tokens = latex_str.split()

    result = []
    for idx, content in enumerate(tokens):
        # 唯一处理：分式命令 → 可视分数线
        if content == r"\frac":
            content = "fraction_line"

        result.append({
            "id": idx + 1,
            "content": content,
            "visible": None,
            "x": None,
            "y": None,
            "w": None,
            "h": None
        })
    return result


def process_caption_file(file_path: str, start_line: int, end_line: int):
    output_dir = "ast"
    os.makedirs(output_dir, exist_ok=True)

    if start_line < 1 or end_line < start_line:
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            all_lines = [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        return

    total_lines = len(all_lines)
    if end_line > total_lines:
        end_line = total_lines

    for line_num in range(start_line - 1, end_line):
        line = all_lines[line_num]
        current_line_num = line_num + 1

        if "\t" not in line:
            continue

        name, latex = line.split("\t", 1)
        name = name.strip()
        latex = latex.strip()

        token_data = latex_to_flat_tokens(latex)

        json_filename = os.path.join(output_dir, f"{name}.json")
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(token_data, f, ensure_ascii=False, indent=4)


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    CAPTION_FILE = "caption_demo.txt"
    START_LINE = 1
    END_LINE = 1
    process_caption_file(CAPTION_FILE, START_LINE, END_LINE)