import os
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['figure.dpi'] = 300


def read_captions(file_path: str) -> list[tuple[str, str]]:
    captions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t', 1)
            if len(parts) != 2:
                print(f"警告：第{line_num}行格式错误，已跳过: {line}")
                continue
            filename, latex_expr = parts
            captions.append((filename.strip(), latex_expr.strip()))
    return captions


def render_latex_to_image(latex_str: str, output_path: str, fontsize: int = 28):
    fig, ax = plt.subplots(figsize=(2, 1))
    ax.set_axis_off()
    ax.text(
        0.5, 0.5,
        f'${latex_str}$',
        fontsize=fontsize,
        ha='center',
        va='center',
        color='black'
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches='tight',
        pad_inches=0.15,
        facecolor='white',
        edgecolor='none'
    )
    plt.close(fig)


def main():
    input_file = r'C:\Users\11321\Desktop\LLMs\HV\data\2019\caption.txt'
    output_dir = r'latex_output_2019'

    os.makedirs(output_dir, exist_ok=True)

    captions = read_captions(input_file)
    print(f"共读取到 {len(captions)} 个LaTeX表达式")

    success = 0
    for filename, latex in captions:
        output_path = os.path.join(output_dir, f"{filename}.png")
        try:
            render_latex_to_image(latex, output_path)
            success += 1
            print(f"[{success}/{len(captions)}] 成功: {filename}.png")
        except Exception as e:
            print(f"失败: {filename} - 错误: {str(e)}")

    print(f"\n渲染完成！成功 {success} 个，失败 {len(captions) - success} 个")
    print(f"图片保存在: {os.path.abspath(output_dir)}")


if __name__ == '__main__':
    main()
