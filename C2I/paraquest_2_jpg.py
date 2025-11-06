import os
import json
import uuid
import pandas as pd
from PIL import Image
from io import BytesIO
from multiprocessing import Pool, cpu_count
import glob
###  把imagenet_1k按照类别标签分类并保存为jpg
PARQUET_DIR = "/storage/v-jinpewang/lab_folder/weiming/datasets/imagenet_1k/data"
# 自动收集该目录下所有 parquet 文件路径
PARQUET_FILES = sorted(glob.glob(os.path.join(PARQUET_DIR, "*.parquet")))

OUTPUT_ROOT = "/storage/v-jinpewang/lab_folder/weiming/datasets/imagenet_1k/temp_storage/3"
TEMP_JSON_DIR = os.path.join(OUTPUT_ROOT, "temp_jsons")
FINAL_JSON_PATH = os.path.join(OUTPUT_ROOT, "image_label_map.json")
RESIZE_SIZE = 256

DEBUG_MODE = False
DEBUG_NUM = 10  # 每个 parquet 仅取前 N 张

NUM_PROCESSES = max(len(PARQUET_FILES), cpu_count()-5)


def process_parquet(parquet_path):
    """处理单个 parquet 文件"""
    print(f"📦 开始处理: {os.path.basename(parquet_path)}")
    df = pd.read_parquet(parquet_path)

    if DEBUG_MODE:
        df = df.head(DEBUG_NUM)
        print(f"⚙️ 调试模式启用，仅处理前 {DEBUG_NUM} 张图片")

    records = []

    for i, row in df.iterrows():
        label = str(row["label"])
        img_bytes = row["image"]["bytes"]

        # 创建标签目录
        label_dir = os.path.join(OUTPUT_ROOT, label)
        os.makedirs(label_dir, exist_ok=True)

        # 保存图片
        img_name = f"{uuid.uuid4()}.jpg"
        img_path = os.path.join(label_dir, img_name)

        try:
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
            img = img.resize((RESIZE_SIZE, RESIZE_SIZE), Image.BICUBIC)
            img.save(img_path, format="JPEG")
        except Exception as e:
            print(f"[WARN] {parquet_path} 第 {i} 张保存失败: {e}")
            continue

        records.append({
            "image_path": img_path,
            "label": label
        })

        if i % 100 == 0:
            print(f"{os.path.basename(parquet_path)} 已处理 {i}/{len(df)} 张图片")

    # 写入临时 JSON 文件
    os.makedirs(TEMP_JSON_DIR, exist_ok=True)
    temp_json_path = os.path.join(TEMP_JSON_DIR, f"{os.path.basename(parquet_path)}.json")
    with open(temp_json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=4)

    print(f"✅ 完成 {parquet_path}，共保存 {len(records)} 张图片")
    return temp_json_path


if __name__ == "__main__":
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(TEMP_JSON_DIR, exist_ok=True)

    print(f"🚀 启动多进程，进程数: {NUM_PROCESSES}")

    with Pool(processes=NUM_PROCESSES) as pool:
        temp_json_files = pool.map(process_parquet, PARQUET_FILES)

    # === 合并所有 JSON ===
    print("\n🧩 正在合并所有临时 JSON...")
    merged_records = []
    for path in temp_json_files:
        with open(path, "r", encoding="utf-8") as f:
            merged_records.extend(json.load(f))

    with open(FINAL_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(merged_records, f, ensure_ascii=False, indent=4)

    print(f"\n🎉 所有任务完成！共保存 {len(merged_records)} 张图片")
    print(f"👉 最终 JSON 文件路径: {FINAL_JSON_PATH}")
