"""
Text-to-Image Dataset Loader

This script loads images and their corresponding prompts from the text-to-image-2M dataset,
filters them based on prompt word count, and saves them locally.
"""

import os
from io import BytesIO
from datasets import load_dataset
from PIL import Image
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm


#### 这里放到output 文件夹中
# ========== Configuration ==========
# Dataset settings
BASE_URL = "https://huggingface.co/datasets/jackyhate/text-to-image-2M/resolve/main/data_512_2M/data_{i:06d}.tar"
NUM_SHARDS = 46  # Number of webdataset tar files
DEBUG_MODE = False   ## 改为 False 时不限制数量
DEBUG_MAX_IMAGES = 100000  ## 测试数量
# Filter and save settings
MAX_WORD_COUNT = 50  # Maximum word count for prompts (skip if exceeded)

# SAVE_DIR = "/storage/v-jinpewang/lab_folder/weiming/exp/temp/junchao/temp_storage/t2i1/load_image"  # Directory to save images and prompts
IMAGE_SAVE_DIR = "/storage/v-jinpewang/lab_folder/weiming/exp/temp/test/t2i1/images_save"
TEXT_SAVE_DIR = "/storage/v-jinpewang/lab_folder/weiming/exp/temp/test/t2i1/texts_save"

os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
os.makedirs(TEXT_SAVE_DIR, exist_ok=True)

# ========== Helper Functions ==========
def extract_image(sample):
    """Extract image from sample data."""
    return sample.get('jpg') or sample.get('image') or sample.get('png')


def extract_prompt(sample):
    """Extract prompt text from sample data."""
    if 'json' in sample and isinstance(sample['json'], dict):
        return sample['json'].get('prompt', 'no_prompt')
    elif 'text' in sample:
        return sample.get('text')
    elif 'caption' in sample:
        return sample.get('caption')
    return 'no_prompt'


def save_image_and_prompt(image, prompt, save_index, save_dir):
    """Save image and prompt to files.
    
    Args:
        image: PIL Image object or bytes
        prompt: Text prompt string
        save_index: Index number for filename
        save_dir: Directory to save files
    """
    # Convert bytes to PIL Image if needed
    if isinstance(image, bytes):
        image = Image.open(BytesIO(image))
    
    # Save image
    image_path = os.path.join(IMAGE_SAVE_DIR, f"image_{save_index:06d}.png")
    image.save(image_path)
    # print(f" Image saved to: {image_path}")
    
    # Save prompt
    prompt_path = os.path.join(TEXT_SAVE_DIR, f"image_{save_index:06d}.txt")
    with open(prompt_path, 'w', encoding='utf-8') as f:
        f.write(str(prompt))
    # print(f" Prompt saved to: {prompt_path}")
    
    # Display prompt preview
    prompt_preview = f"{prompt[:100]}..." if len(str(prompt)) > 100 else prompt
    # print(f"Prompt: {prompt_preview}")
    image.close()


def load_text_to_image_dataset(base_url, num_shards):
    """Load text-to-image dataset from webdataset tar files.
    
    Args:
        base_url: URL template for dataset files
        num_shards: Number of tar file shards
        
    Returns:
        Streaming dataset object
    """
    print(f"Loading dataset from {num_shards} shards...")
    urls = [base_url.format(i=i) for i in range(num_shards)]
    dataset = load_dataset(
        "webdataset",
        data_files={"train": urls},
        split="train",
        streaming=True
    )
    return dataset

def save_image_and_prompt_wrapper(args, save_dir):
    """Wrapper to unpack tuple for multiprocessing."""
    idx, (image, prompt) = args
    try:
        save_image_and_prompt(image, prompt, idx, save_dir)
    except Exception as e:
        print(f"⚠️ Error saving sample {idx}: {e}")


def process_dataset(dataset, max_word_count, save_dir):
    """
    Stream dataset, filter by prompt length, and save images in parallel as they arrive.
    """
    saved_count = 0
    num_processes = max(cpu_count()-10, 10)   ### 内存不够的话小一些

    print("\n🚀 Streaming samples (saving in parallel)...")

    # 启动多进程池
    pool = Pool(processes=num_processes)
    pbar = tqdm(desc="Processing samples", dynamic_ncols=True, total=None)

    for idx, sample in enumerate(dataset):
        image = extract_image(sample)
        prompt = extract_prompt(sample)

        if image is None:
            continue

        word_count = len(str(prompt).split())
        if word_count > max_word_count:
            continue

        # ✅ 每读一个样本就异步提交保存，不再累积到内存
        pool.apply_async(
            save_image_and_prompt_wrapper,
            args=((saved_count, (image, prompt)), save_dir)
        )

        saved_count += 1
        pbar.update(1)

        # ✅ 调试模式快速退出
        if DEBUG_MODE and saved_count >= DEBUG_MAX_IMAGES:
            print(f"\n🧩 Debug mode reached {DEBUG_MAX_IMAGES} samples, stopping early.")
            break

    # ✅ 关闭多进程池并等待所有任务结束
    pool.close()
    pool.join()
    pbar.close()

    print(f"\n✅ All done. Total saved: {saved_count}")
    return saved_count

def main():
    """Main function to load, filter and save text-to-image dataset."""
    # Create save directory
    os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
    print(f"Images will be saved to: {IMAGE_SAVE_DIR}\n")
    os.makedirs(TEXT_SAVE_DIR, exist_ok=True)
    print(f"Prompt will be saved to: {TEXT_SAVE_DIR}\n")
    

    # Load dataset
    dataset = load_text_to_image_dataset(BASE_URL, NUM_SHARDS)
    
    # Process and save images
    saved_count = process_dataset(
        dataset,
        max_word_count=MAX_WORD_COUNT,
        save_dir=None
    )
    
    print(f"\n Done! Total saved: {saved_count} images")


if __name__ == "__main__":
    main()

