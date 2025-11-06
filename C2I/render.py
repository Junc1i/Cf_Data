import os
import glob
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from multiprocessing import Pool, cpu_count
from functools import partial
import json
import re
from functools import partial


# 配置
FONT_PATH = ""
IMAGE_SIZE = 512
NUM_PROCESSES = min(35, cpu_count())  # 使用的进程数，最多16个

FONT_CACHE = {}

def get_font(font_path, size):
    key = (font_path, size)
    if key not in FONT_CACHE:
        FONT_CACHE[key] = ImageFont.truetype(font_path, size)
    return FONT_CACHE[key]


BREAK_CHARS = r"_\-/\.\|\\"

CAMEL_RE = re.compile(r'(?<=[a-z0-9])(?=[A-Z])')

def measure_width(draw, font, s: str) -> int:
    if not s:
        return 0
    x0, y0, x1, y1 = draw.textbbox((0, 0), s, font=font)
    return x1 - x0

def split_camel(s: str):
    # 在驼峰边界插入空格再切
    return CAMEL_RE.sub(' ', s).split(' ')

def smart_tokenize(s: str):
    """
    智能分词：
    1) 先按分隔符拆开，并把分隔符作为独立token保留（用于优先换行）
    2) 再对每个非空token做一次驼峰拆分
    """
    if not s:
        return []
    # 先按分隔符拆，保留分隔符
    parts = re.split(f'([{BREAK_CHARS}])', s)
    tokens = []
    for part in parts:
        if part == '':
            continue
        if len(part) == 1 and re.match(f'[{BREAK_CHARS}]', part):
            # ✅ 下划线替换为空格，不保留
            if part == '_':
                tokens.append(' ')
            else:
                tokens.append(part)
        else:
            for seg in split_camel(part):
                if seg != '':
                    tokens.append(seg)
    return tokens


def split_long_token_hard(draw, font, token, max_width):
    """
    硬切策略（仅在极端情况下使用）：
    - 仅当整个单词比 max_width 还宽时才拆
    - 普通单词不拆开（保持整体）
    """
    tmp_img = Image.new("RGB", (1, 1))
    tmp_draw = ImageDraw.Draw(tmp_img)
    width = measure_width(tmp_draw, font, token)

    # ✅ 如果单词整体宽度小于 max_width，不拆
    if width <= max_width:
        return [token]

    # ✅ 若单词整体太长但至少能显示一个字符，则尝试逐字符切
    # （这种情况几乎只发生在超长英文单词或无空格字符串）
    avg_char_width = measure_width(tmp_draw, font, "W")
    if width < 1.5 * max_width or avg_char_width * len(token) < 1.5 * max_width:
        return [token]

    # ✅ 否则强制逐字符切（兜底）
    out, cur = [], ""
    for ch in token:
        if measure_width(tmp_draw, font, cur + ch) <= max_width or not cur:
            cur += ch
        else:
            out.append(cur)
            cur = ch
    if cur:
        out.append(cur)
    return out


def wrap_line(draw, font, text, max_width):
    """
    折行算法（优先软断点与驼峰）：
    - 先用 smart_tokenize 得到 token 流（含分隔符独立token）
    - 测试追加 token 后是否超宽；若超宽且 token 本身超宽，调用硬切
    - 分隔符 token 若导致超宽，将其放到下一行开头（避免悬挂到行末）
    """
    if text == "":
        return [""]
    # print(f"[WRAP] text='{text}', max_width={max_width}")
    tokens = smart_tokenize(text)
    lines = []
    cur = ""

    for tok in tokens:
        candidate = (cur + tok) if (cur == "" or cur.endswith(" ") or len(tok) == 1 and re.match(f'[{BREAK_CHARS}]', tok)) else (cur + " " + tok)
        w = measure_width(draw, font, candidate)

        if w <= max_width:
            cur = candidate
        else:
            # 先收当前行（若有）
            if cur != "":
                lines.append(cur)
                cur = ""

            # 当前 token 自己超宽 -> 硬切后逐段放入
            if measure_width(draw, font, tok) > max_width:
                # ✅ 若是普通英文单词，整块换行，不拆
                if re.match(r'^[A-Za-z0-9]+$', tok):
                    if cur:
                        lines.append(cur.strip())
                    cur = tok
                    continue
                for piece in split_long_token_hard(draw, font, tok, max_width):
                    if measure_width(draw, font, piece) <= max_width:
                        if cur == "":
                            cur = piece
                        else:
                            cand2 = (cur + " " + piece)
                            if measure_width(draw, font, cand2) <= max_width:
                                cur = cand2
                            else:
                                lines.append(cur)
                                cur = piece
                    else:
                        # 极端情况：字符本身超宽（几乎不会发生），强制成独行
                        if cur:
                            lines.append(cur)
                        lines.append(piece)
                        cur = ""
            else:
                # token 不超宽但与当前行合并超宽 -> 放到新行
                cur = tok

    if cur != "":
        lines.append(cur)

    # 去除行首尾多余空格（可选）
    lines = [re.sub(r'\s+', ' ', ln.strip()) for ln in lines]
    # print(f"[WRAP] lines={lines}")
    lines = [ln.strip() for ln in lines]
    
    return lines

def layout_for_font_size(text, font_path, font_size, image_size, margin, line_gap):
    font = get_font(font_path, font_size)
    tmp = Image.new("RGB", (image_size, image_size))
    draw = ImageDraw.Draw(tmp)

    max_width = image_size - 2 * margin
    max_height = image_size - 2 * margin

    paragraphs = text.split("\n")
    lines = []
    for para in paragraphs:
        wrapped = wrap_line(draw, font, para, max_width)
        if not wrapped:
            wrapped = [""]
        lines.extend(wrapped)

    ascent, descent = font.getmetrics()
    line_height = ascent + descent

    total_height = 0
    for i in range(len(lines)):
        total_height += line_height
        if i != len(lines) - 1:
            total_height += line_gap

    fits = total_height <= max_height
    # print(f"[LAYOUT] font={font_size}, lines={len(lines)}, total_height={total_height}, "
    #       f"max_height={max_height}, fits={fits}, first_line='{lines[0] if lines else ''}'")
    return fits, lines, font, line_height, total_height

def render_text_with_adaptive_font(
    text, font_path, image_size, text_color, background_color, margin=10, line_gap=4
):
    left, right = 10, 200
    best = None
    
    while left <= right:
        mid = (left + right) // 2
        try:
            fits, lines, font, line_height, total_height = layout_for_font_size(
                text, font_path, mid, image_size, margin, line_gap
            )
            
        except Exception:
            fits = False

        if fits:
            best = (mid, lines, font, line_height, total_height)
            left = mid + 1
        else:
            right = mid - 1

    if best is None:
        raise ValueError(f"Cannot render text: {text[:80]}...")

    # 最终兜底检查（宽/高）
    while True:
        font_size, lines, font, line_height, total_height = best
        img = Image.new("RGB", (image_size, image_size), color=background_color)
        draw = ImageDraw.Draw(img)
        font = get_font(font_path, font_size)

        max_width = image_size - 2 * margin
        max_height = image_size - 2 * margin

        too_wide = any(measure_width(draw, font, line) > max_width for line in lines)
        too_tall = total_height > max_height

        if too_wide or too_tall:
            font_size -= 1
            if font_size < 10:
                break
            try:
                fits, lines, font, line_height, total_height = layout_for_font_size(
                    text, font_path, font_size, image_size, margin, line_gap
                )
                

                if not fits:
                    continue
                best = (font_size, lines, font, line_height, total_height)
                continue
            except Exception:
                continue

        # 绘制（垂直居中）
        y = margin + (max_height - total_height) // 2
        x = margin
        for i, line in enumerate(lines):
            # print(f"  line{i} width={measure_width(draw, font, line)} text='{line}'")
            draw.text((x, y), line, fill=text_color, font=font)
            y += line_height
            if i != len(lines) - 1:
                y += line_gap

        return img

    raise ValueError("Text rendering failed after fallback.")

def process_item(entry, label_map, output_root):
    try:
        image_path = entry["image_path"]
        label = str(entry["label"])
        name = label_map.get(label, None)
        if not name:
            # print(f"⚠️ Label {label} not found in label_map.")
            return 0

        image = render_text_with_adaptive_font(
            text=name,
            font_path=FONT_PATH,
            image_size=IMAGE_SIZE,
            background_color=(255, 255, 255),
            text_color=(0, 0, 0),
            margin=20
        )

        label_dir = os.path.join(output_root, label)
        os.makedirs(label_dir, exist_ok=True)

        base_name = os.path.basename(image_path)
        save_name = os.path.splitext(base_name)[0] + "_render.JPEG"
        save_path = os.path.join(label_dir, save_name)
        image.save(save_path, format="JPEG")
        return 1

    except Exception as e:
        # print(f"⚠️ Error rendering {entry}: {e}")
        return 0


def main():
    print(f"🎨 Starting text rendering process...")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Font: {FONT_PATH}")
    print(f"🚀 Using {NUM_PROCESSES} processes\n")

    image_json_path = "./image_label_map.json"  # paraquest_2_jpg.py 输出的json位置
    label_json_path = "./label_map.json"   # 压缩包中的label_map.json文件位置
    output_root = ""   ### render图片的保存位置
    os.makedirs(output_root, exist_ok=True)

    # 读取 JSON
    with open(image_json_path, 'r', encoding='utf-8') as f:
        image_list = json.load(f)
    with open(label_json_path, 'r', encoding='utf-8') as f:
        label_data = json.load(f)

    # 兼容两种格式
    if isinstance(label_data, list):
        label_map = {str(item["label"]): item["name"] for item in label_data}
    elif isinstance(label_data, dict):
        label_map = {str(k): v for k, v in label_data.items()}
    else:
        raise ValueError("Unsupported label_json format")

    # print(f"📦 Loaded {len(image_list)} image entries and {len(label_map)} labels.\n")

    # 绑定额外参数
    worker_func = partial(process_item, label_map=label_map, output_root=output_root)

    # 并行渲染
    total_success = 0
    with Pool(processes=NUM_PROCESSES) as pool:
        for ok in tqdm(pool.imap_unordered(worker_func, image_list),
                       total=len(image_list), desc="Rendering"):
            total_success += ok

    print(f"\n🎉 All done! Rendered {total_success}/{len(image_list)} images.")
    print(f"📂 Saved under: {output_root}")

if __name__ == "__main__":
    # 测试模式：先生成几张示例图片查看效果
    # 如果效果满意，将下面的 test_mode() 改为 main() 即可正式处理所有数据
    # test_mode()
    
    # 正式处理所有数据（测试满意后取消注释）
    main()

"""

FONT_PATH改为字体位置

修改main函数中的
image_json_path     paraquest_2_jpg.py 输出的json位置
label_json_path     压缩包中的label_map.json文件位置
output_root 换为render结果图保存位置
"""