import matplotlib.pyplot as plt
import os
from skimage.metrics import structural_similarity as ssim
import numpy as np
from PIL import Image

# -------------------------- 配置参数（文件名极简） --------------------------
IMAGE_SAVE_DIR = "latex_formula_images"
LATEX_LIST = [
    r"x^2 + y_3 = 10",
    r"x^{2} + y_{3} = 10",
    r"(a)",
    r"(\alpha)",
    r"\int_0^\infty",
    r"\int\limits_0^\infty",
    r"\sqrt[3]{xyz} + 12 = 20",
    r"m \div n + p \pm q = r",
    r"a^2 + b^2 \equiv c^2",
    r"5.2 \times 3.8 \div 1.9 = 10.4",
    r"\sqrt{xa} \pm \sqrt[2]{zw} \neq 0",
    r"\sqrt { \frac { 2 z ^ { 3 } } { \sqrt { \frac { 3 z ^ { 2 } } { \sqrt { 4 z } } } } }",
    r"\pi \int \limits _ { c } ^ { d } \{ g ( y ) \} ^ { 2 } d y",
    r"( \frac { 1 + x ^ { 2 } } { 1 + y ^ { 2 } } ) ^ { k } \leq 2 ^ { | t | } ( 1 + ( x - y ) ^ { 2 } ) ^ { | t | }",
    r"g ( x , y ) = \sqrt [ 3 ] { x - y } + \sqrt { | x + y | }",
    r"\sum \limits _ { n = 1 } ^ { N } ( - 1 ) ^ { n } \sin ( n x )",
    r"f ^ { ( n ) } ( a )",
    r"3 8 + 6 1 \leq 9 9",
    r"\frac{3}{5}"
]
DPI = 300
FONT_SIZE = 24

os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

# -------------------------- 核心函数：背景贴合+文件名简化 --------------------------
def latex_to_image(latex: str, save_path: str) -> None:
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'Computer Modern'
    plt.rcParams['figure.dpi'] = DPI
    plt.rcParams['text.latex.preamble'] = r'''
        \usepackage{amsmath}
        \usepackage{amssymb}
        \usepackage[utf8]{inputenc}
    '''

    # 背景贴合公式（无多余空白）
    fig = plt.figure(figsize=(1, 1))
    ax = fig.add_axes([0, 0, 1, 1])  # 占满画布
    ax.axis('off')

    text_obj = ax.text(0.5, 0.5, f"${latex}$", fontsize=FONT_SIZE, ha='center', va='center')

    # 动态调整画布尺寸
    fig.canvas.draw()
    bbox = text_obj.get_window_extent()
    width = bbox.width / DPI
    height = bbox.height / DPI
    fig.set_size_inches(width, height)

    # 保存（极窄边距）
    fig.savefig(
        save_path,
        bbox_inches='tight',
        pad_inches=0.1,
        format='png',
        facecolor='white'
    )
    plt.close()
    print(f"✅ 生成图像：{os.path.basename(save_path)}（尺寸：{width:.2f}×{height:.2f}英寸）")

# -------------------------- 批量生成（文件名仅序号） --------------------------
def batch_generate_images():
    print("开始生成【背景贴合+文件名简化】的LaTeX图像...\n")
    for idx, latex in enumerate(LATEX_LIST, 1):
        # 文件名简化：仅保留序号 → formula_1.png, formula_2.png...
        img_filename = f"formula_{idx}.png"
        img_save_path = os.path.join(IMAGE_SAVE_DIR, img_filename)
        latex_to_image(latex, img_save_path)
    print(f"\n🎉 所有图像生成完成！")
    print(f"📁 保存路径：{IMAGE_SAVE_DIR}")
    print(f"📌 文件名格式：formula_1.png ~ formula_10.png（共10个）")


def normalize_img(path, target_h=128, canvas=(512, 256)):
    """裁白边 -> 等比缩放到统一高度 -> 居中贴到统一画布"""
    img = Image.open(path).convert("L")
    arr = np.array(img)
    # 1. 裁掉白边（找非白像素的最小包围框）
    mask = arr < 250
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.full(canvas[::-1], 255, dtype=np.uint8)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    crop = img.crop((x0, y0, x1, y1))
    # 2. 等比缩放到统一高度
    w, h = crop.size
    new_w = max(1, int(w * target_h / h))
    crop = crop.resize((new_w, target_h))
    # 3. 居中贴到统一画布
    cw, ch = canvas
    bg = Image.new("L", canvas, 255)
    paste_x = max(0, (cw - new_w) // 2)
    bg.paste(crop, (paste_x, (ch - target_h) // 2))
    return np.array(bg)

def is_same_formula(path1, path2, thr=0.945):
    a = normalize_img(path1)
    b = normalize_img(path2)
    score = ssim(a, b)
    print(f"SSIM = {score:.4f} -> {'同一公式' if score >= thr else '不同公式'}")
    return score >= thr

if __name__ == "__main__":
    batch_generate_images()
    is_same_formula(r".\latex_formula_images\formula_1.png", r".\latex_formula_images\formula_2.png")
    is_same_formula(r".\latex_formula_images\formula_3.png", r".\latex_formula_images\formula_4.png")
    is_same_formula(r".\latex_formula_images\formula_5.png", r".\latex_formula_images\formula_4.png")
    is_same_formula(r".\latex_formula_images\formula_11.png", r".\latex_formula_images\formula_12.png")
