import os
import json
import numpy as np
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# -------------------------- 配置参数 --------------------------
INPUT_PATH = r"C:\Users\11321\Desktop\LLMs\HG\Main\indicator\box\2009213-137-85.json"
OUTPUT_DIR = "./box/bbox_images"

SQ_IMG_PATH = r"./box/sq.png"

DRAW_BOX = True          # 是否绘制红色方框
DRAW_SYMBOL = True       # 是否在框内绘制印刷体符号
DRAW_LABEL = False       # 是否在框上方标注文字内容

PADDING = 20
BOX_COLOR = "red"
BOX_WIDTH = 2
LABEL_COLOR = "blue"
LABEL_FONT_SIZE = 16
SYMBOL_DPI = 300

_symbol_cache = {}

def load_sqrt_transparent() -> Image.Image:
    cache_key = "__sqrt_local__"
    if cache_key in _symbol_cache:
        return _symbol_cache[cache_key].copy()

    img = Image.open(SQ_IMG_PATH).convert("L")
    arr = np.array(img)
    alpha = 255 - arr
    rgba_img = Image.new("RGBA", img.size, (0, 0, 0, 0))
    rgba_arr = np.array(rgba_img)
    rgba_arr[:, :, 3] = alpha
    rgba_arr[:, :, 0:3] = 0

    result_img = Image.fromarray(rgba_arr)
    _symbol_cache[cache_key] = result_img
    return result_img.copy()


def render_latex_symbol(content: str) -> Image.Image:
    if content in _symbol_cache:
        return _symbol_cache[content].copy()

    if content == r"\sqrt":
        return load_sqrt_transparent()

    elif content == "fraction_line":
        line_img = Image.new("RGBA", (100, 2), (0, 0, 0, 255))
        _symbol_cache[content] = line_img
        return line_img.copy()

    else:
        latex_code = content

    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'Computer Modern'
    plt.rcParams['text.latex.preamble'] = r'''
        \usepackage{amsmath}
        \usepackage{amssymb}
        \usepackage[utf8]{inputenc}
    '''

    fig = plt.figure(figsize=(2, 2), dpi=SYMBOL_DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    try:
        ax.text(0.5, 0.5, f"${latex_code}$", fontsize=40,
                ha='center', va='center', color='black')
    except Exception as e:
        print(f"⚠️  符号渲染失败: {content} | {str(e)}")
        plt.close(fig)
        empty = Image.new("RGBA", (10, 10), (255, 255, 255, 0))
        _symbol_cache[content] = empty
        return empty

    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight',
                pad_inches=0, transparent=True, dpi=SYMBOL_DPI)
    plt.close(fig)
    buf.seek(0)

    img = Image.open(buf).convert("RGBA")

    # 按alpha通道裁剪多余透明边
    alpha = np.array(img.split()[-1])
    if alpha.max() == 0:
        _symbol_cache[content] = img
        return img.copy()

    coords = np.argwhere(alpha > 0)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    img = img.crop((x0, y0, x1, y1))

    _symbol_cache[content] = img
    return img.copy()


def draw_bboxes_from_json(json_path: str, save_path: str) -> None:
    with open(json_path, 'r', encoding='utf-8') as f:
        symbols = json.load(f)

    visible_symbols = [s for s in symbols if s.get("visible", False)]
    if not visible_symbols:
        print(f"⚠️  {os.path.basename(json_path)} 无可见符号，生成空白图")
        Image.new("RGB", (200, 100), "white").save(save_path)
        return

    max_x = max(int(s["x"]) + int(s["w"]) for s in visible_symbols)
    max_y = max(int(s["y"]) + int(s["h"]) for s in visible_symbols)
    canvas_w = max_x + PADDING * 2
    canvas_h = max_y + PADDING * 2

    img = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(img)

    try:
        label_font = ImageFont.truetype("arial.ttf", LABEL_FONT_SIZE)
    except Exception:
        label_font = ImageFont.load_default()

    for sym in visible_symbols:
        x = int(sym["x"]) + PADDING
        y = int(sym["y"]) + PADDING
        w = int(sym["w"])
        h = int(sym["h"])
        content = sym.get("content", "")

        if DRAW_SYMBOL:
            sym_img = render_latex_symbol(content)
            sym_img = sym_img.resize((w, h), Image.Resampling.LANCZOS)
            img.paste(sym_img, (x, y), mask=sym_img)

        if DRAW_BOX:
            draw.rectangle(
                xy=[x, y, x + w, y + h],
                outline=BOX_COLOR,
                width=BOX_WIDTH
            )

        if DRAW_LABEL:
            text_y = max(2, y - LABEL_FONT_SIZE - 2)
            draw.text((x, text_y), content, fill=LABEL_COLOR, font=label_font)

    img.save(save_path)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.isfile(INPUT_PATH) and INPUT_PATH.endswith(".json"):
        json_files = [INPUT_PATH]
    elif os.path.isdir(INPUT_PATH):
        json_files = [
            os.path.join(INPUT_PATH, f)
            for f in os.listdir(INPUT_PATH)
            if f.endswith(".json")
        ]
    else:
        print("输入路径无效，请检查 INPUT_PATH 配置")
        return

    if not json_files:
        print("未找到任何JSON文件")
        return

    print(f"共找到 {len(json_files)} 个JSON文件，开始绘制...\n")

    for idx, json_path in enumerate(json_files, 1):
        file_name = os.path.basename(json_path)
        stem = os.path.splitext(file_name)[0]
        save_path = os.path.join(OUTPUT_DIR, f"{stem}_bbox.png")

        print(f"[{idx}/{len(json_files)}] 处理: {file_name}")
        try:
            draw_bboxes_from_json(json_path, save_path)
            print(f"已保存: {os.path.basename(save_path)}")
        except Exception as e:
            print(f"处理失败: {e}")

    print(f"\n全部完成！输出文件夹：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()