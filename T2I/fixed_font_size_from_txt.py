import os
import glob
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from multiprocessing import Pool, cpu_count
from functools import partial
import json
import re

# 配置
FONT_PATH = "/storage/v-jinpewang/lab_folder/weiming/test_set/open-sans/OpenSans-Regular.ttf"
IMAGE_SIZE = 512  ## 分辨率   v-jinpewang/lab_folder/weiming/exp/temp/test/t2i1/images_save_input/image_000041.JPEG
OUTPUT_DIR = "/storage/v-jinpewang/lab_folder/weiming/exp/temp/test/t2i1/images_save_input/"   ### 保存位置
TEXT_INPUT_DIR = "/storage/v-jinpewang/lab_folder/weiming/exp/temp/test/t2i1/texts_save/"   ### txt位置
NUM_PROCESSES = max(10, cpu_count()-2)  # 使用的进程数，
# NUM_PROCESSES = 2




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

def render_from_text(args):
    text, save_path = args
    try:
        img = render_text_with_adaptive_font(
            text=text,
            font_path=FONT_PATH,
            image_size=IMAGE_SIZE,
            background_color=(255, 255, 255),
            text_color=(0, 0, 0),
            margin=20
        )
        img.save(save_path, format="JPEG")
        return (True, save_path)
    except Exception as e:
        return (False, str(e))


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
    font = ImageFont.truetype(font_path, font_size)
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


def process_single_file(json_path_pair, render_dir):
    """
    json_path_pair 是一个 (cls_json_path, img_json_path) 二元组
    渲染每个图片对应的类别文字为图片
    """
    try:
        cls_json_path, img_json_path = json_path_pair

        # 1️⃣ 读取两个JSON
        with open(cls_json_path, 'r', encoding='utf-8') as f:
            cls_data = json.load(f)  # {"0": ["n01440764", "tench"], ...}

        with open(img_json_path, 'r', encoding='utf-8') as f:
            img_data = json.load(f)  # {"n01440764": ["n01440764_10043.JPEG", ...]}

        success_count = 0
        error_count = 0

        # 2️⃣ 遍历类别映射
        for _, (cls_id, cls_name) in cls_data.items():
            if cls_id not in img_data:
                continue

            img_list = img_data[cls_id]
            if not img_list:
                continue

            # 为该类别创建输出目录
            cls_render_dir = os.path.join(render_dir, cls_id)
            os.makedirs(cls_render_dir, exist_ok=True)

            # 3️⃣ 遍历该类下所有图片
            for img_file in img_list:
                try:
                    text = str(cls_name).strip()
                    if not text:
                        continue

                    # 渲染类别文字
                    image = render_text_with_adaptive_font(
                        text=text,
                        font_path=FONT_PATH,
                        image_size=IMAGE_SIZE,
                        background_color=(255, 255, 255),
                        text_color=(0, 0, 0),
                        margin=20
                    )

                    # 4️⃣ 保存文件
                    base_name = os.path.splitext(img_file)[0]
                    save_name = f"{base_name}_render.JPEG"
                    save_path = os.path.join(cls_render_dir, save_name)
                    image.save(save_path, format="JPEG")

                    success_count += 1

                except Exception as sub_e:
                    error_count += 1
                    print(f"⚠️ Error rendering {img_file}: {sub_e}")

        if success_count == 0:
            return (False, f"No valid renders in {img_json_path}")

        return (True, f"{success_count} rendered, {error_count} errors")

    except Exception as e:
        return (False, f"Error processing {json_path_pair}: {e}")




def main():
    print(f"🎨 Starting text rendering process...")

    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Font: {FONT_PATH}")
    print(f"🚀 Using {NUM_PROCESSES} processes\n")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    txt_files = sorted(glob.glob(os.path.join(TEXT_INPUT_DIR, "*.txt")))
    print(f"📦 Found {len(txt_files)} text files to render.\n")
    tasks = []
    for txt_path in txt_files:
        file_name = os.path.basename(txt_path)
        save_name = os.path.splitext(file_name)[0] + ".JPEG"
        save_path = os.path.join(OUTPUT_DIR, save_name)
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        tasks.append((text, save_path))
    
    total_success, total_error = 0, 0

    with Pool(processes=NUM_PROCESSES) as pool:
        results = list(tqdm(
            pool.imap_unordered(render_from_text, tasks),
            total=len(tasks),
            desc="Rendering text files"
        ))
    for success, msg in results:
        if success:
            total_success += 1
        else:
            total_error += 1
            print(f"⚠️ Error: {msg}")

    print(f"\n🎉 All done!")
    print(f"✅ Total rendered: {total_success}")
    print(f"⚠️ Failed: {total_error}")
    print(f"📂 Output saved under: {OUTPUT_DIR}")

    


if __name__ == "__main__":
    main()

