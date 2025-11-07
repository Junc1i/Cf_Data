import os
import json
import uuid
import pandas as pd
from PIL import Image
from io import BytesIO
from multiprocessing import Pool, cpu_count
import glob
from multiprocessing import Manager

###  把imagenet_1k按照类别标签分类并保存为jpg
PARQUET_DIR = "/storage/v-jinpewang/lab_folder/weiming/datasets/imagenet_1k/data"
# 自动收集该目录下所有 parquet 文件路径
# all_parquet_files = glob.glob(os.path.join(PARQUET_DIR, "*.parquet"))
# PARQUET_FILES = sorted(
#     [p for p in all_parquet_files if "validation-" not in os.path.basename(p)]
# )

PARQUET_FILES = sorted(glob.glob(os.path.join(PARQUET_DIR, "*.parquet")))



OUTPUT_ROOT = "/storage/v-jinpewang/lab_folder/weiming/datasets/imagenet_1k/temp_storage/test_5"
TEMP_JSON_DIR = os.path.join(OUTPUT_ROOT, "temp_jsons")
FINAL_JSON_PATH = os.path.join(OUTPUT_ROOT, "image_label_map.json")
RESIZE_SIZE = 256

DEBUG_MODE = False
DEBUG_NUM = 10  # 每个 parquet 仅取前 N 张

# NUM_PROCESSES = max(len(PARQUET_FILES), cpu_count()-5)
NUM_PROCESSES = max(1, min(len(PARQUET_FILES), cpu_count() - 2))
MAX_PER_CLASS = 1000          # 🔹 每类最多50张
MAX_COMPLETED_CLASSES = 1000 # 🔹 满足1000类就停止整个程序


def process_parquet(parquet_path, shared_completed_labels, shared_counts, lock):
    """处理单个 parquet 文件"""
    if len(shared_completed_labels) >= MAX_COMPLETED_CLASSES:
        print("🚫 达到类别上限，跳过", parquet_path)
        return None

    print(f"📦 开始处理: {os.path.basename(parquet_path)}")
    df = pd.read_parquet(parquet_path)

    if DEBUG_MODE:
        df = df.head(DEBUG_NUM)
        print(f"⚙️ 调试模式启用，仅处理前 {DEBUG_NUM} 张图片")

    records = []
    label_counters = {}
    for i, row in df.iterrows():
        # 先取出本行数据
        label_value = row["label"]
        if label_value == -1:   # 🚫 跳过 label == -1 的样本
            continue
        label = str(label_value)
        img_bytes = row["image"]["bytes"]

        # === 加锁：全局类数与该类名额“预占位” ===
        with lock:
            # 达到全局1000类就收工
            if len(shared_completed_labels) >= MAX_COMPLETED_CLASSES:
                print("✅ 已达到1000个类别上限，提前结束进程")
                break

            # 若该类已完成，直接跳过
            if label in shared_completed_labels:
                continue

            # 当前已保存数（全局）
            curr = shared_counts.get(label, 0)
            if curr >= MAX_PER_CLASS:
                # 第一次到达500时登记完成
                if label not in shared_completed_labels:
                    shared_completed_labels.append(label)
                    print(f"🏁 类别 {label} 已收集满 {MAX_PER_CLASS} 张，共完成 {len(shared_completed_labels)} 类")
                continue

            # 预占一个名额（防止并发超额）
            shared_counts[label] = curr + 1
            reserved_to = curr + 1

        # === 无锁区：实际写文件 ===
        label_dir = os.path.join(OUTPUT_ROOT, label)
        os.makedirs(label_dir, exist_ok=True)

        img_name = f"{uuid.uuid4()}.jpg"
        img_path = os.path.join(label_dir, img_name)

        try:
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
            # img = img.resize((RESIZE_SIZE, RESIZE_SIZE), Image.BICUBIC)
            img.save(img_path, format="JPEG")
        except Exception as e:
            print(f"[WARN] {parquet_path} 第 {i} 张保存失败: {e}")
            # 回滚预占名额
            with lock:
                shared_counts[label] = max(0, shared_counts.get(label, 1) - 1)
            continue

        # 成功后记录
        records.append({"image_path": img_path, "label": label})

        # 若正好达500，登记完成（只在成功保存后登记）
        if reserved_to == MAX_PER_CLASS:
            with lock:
                if label not in shared_completed_labels and shared_counts.get(label, 0) >= MAX_PER_CLASS:
                    shared_completed_labels.append(label)
                    print(f"🏁 类别 {label} 已收集满 {MAX_PER_CLASS} 张，共完成 {len(shared_completed_labels)} 类")

        if i % 100 == 0:
            print(f"{os.path.basename(parquet_path)} 已处理 {i}/{len(df)} 张图片")
    if not records:
        print(f"ℹ️ {parquet_path} 本批无可保存记录")
        return None
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
    manager = Manager()
    shared_completed_labels = manager.list()
    shared_counts = manager.dict() 
    lock = manager.Lock()
    with Pool(processes=NUM_PROCESSES) as pool:
        temp_json_files = pool.starmap(
            process_parquet,
            [(p, shared_completed_labels, shared_counts, lock) for p in PARQUET_FILES]
        )
    # with Pool(processes=NUM_PROCESSES) as pool:
    #     temp_json_files = pool.map(process_parquet, PARQUET_FILES)

    # === 合并所有 JSON ===
    print("\n🧩 正在合并所有临时 JSON...")
    merged_records = []
    for path in filter(None, temp_json_files):
        with open(path, "r", encoding="utf-8") as f:
            merged_records.extend(json.load(f))

    print("\n🧹 检查并修正每个类别图片数量...")
    from glob import glob
    valid_paths = set()  # 存放保留的图片路径

    for label in os.listdir(OUTPUT_ROOT):
        label_dir = os.path.join(OUTPUT_ROOT, label)
        if not os.path.isdir(label_dir):
            continue
        imgs = sorted(glob(os.path.join(label_dir, "*.jpg")))
        if len(imgs) > MAX_PER_CLASS:
            extra = imgs[MAX_PER_CLASS:]
            for p in extra:
                os.remove(p)
            print(f"⚖️ {label}: 超出 {len(imgs) - MAX_PER_CLASS} 张，已删除多余图片")
            imgs = imgs[:MAX_PER_CLASS]
        for p in imgs:
            valid_paths.add(os.path.abspath(p))

    # 过滤 JSON 中无效图片路径
    before = len(merged_records)
    merged_records = [rec for rec in merged_records if os.path.abspath(rec["image_path"]) in valid_paths]
    after = len(merged_records)
    print(f"🧾 JSON 修正完成，移除 {before - after} 条无效记录")
        # === 检查并修正超限类别 ===
    from collections import Counter
    from glob import glob

    label_counts = Counter([rec["label"] for rec in merged_records])
    over_labels = {k: v for k, v in label_counts.items() if v > MAX_PER_CLASS}

    if over_labels:
        print("\n🚨 检测到以下类别超过限制，开始修正...")
        valid_paths = set(p["image_path"] for p in merged_records)  # 当前JSON中的所有路径
        removed_paths = set()

        for label, count in sorted(over_labels.items(), key=lambda x: -x[1]):
            label_dir = os.path.join(OUTPUT_ROOT, label)
            imgs = sorted(glob(os.path.join(label_dir, "*.jpg")))
            if len(imgs) > MAX_PER_CLASS:
                extra = imgs[MAX_PER_CLASS:]
                for p in extra:
                    try:
                        os.remove(p)
                        removed_paths.add(os.path.abspath(p))
                    except Exception as e:
                        print(f"[WARN] 删除 {p} 失败: {e}")
                print(f"⚖️ {label}: 已删除 {len(extra)} 张多余图片")

        # 同步修正 JSON
        before = len(merged_records)
        merged_records = [rec for rec in merged_records if os.path.abspath(rec["image_path"]) not in removed_paths]
        after = len(merged_records)
        print(f"✅ 已移除 {before - after} 条超限记录")
    else:
        print("\n✅ 没有类别超过 MAX_PER_CLASS 限制")
    # === 写出最终 JSON ===
    with open(FINAL_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(merged_records, f, ensure_ascii=False, indent=4)

    print(f"\n🎉 所有任务完成！共保存 {len(merged_records)} 张图片")
    print(f"👉 最终 JSON 文件路径: {FINAL_JSON_PATH}")
