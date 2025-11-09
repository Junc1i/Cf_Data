"""
This file is used to extract feature every iteration interval for saving checkpoints, testing FID and visual testing in config.sample.path directory.
"""

import os
import shutil
import sys
# 往上两级到 Crossflow 目录
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import json
import argparse
from transformers import AutoModelForCausalLM

print("sys.path:",sys.path)
original_sys_path = sys.path.copy()
crossflow_parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
janus_dir = os.path.join(crossflow_parent_dir, "Janus")
# sys.path.insert(0, os.path.abspath(janus_dir))
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
sys.path = original_sys_path  # 直接恢复

def recreate_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)


def main(bz=128, device="cuda:7", image_dir=None, save_dir=None):
    # 使用默认值（如果没有通过参数传入）
    if image_dir is None:
        image_dir = '/storage/v-jinpewang/lab_folder/junchao/crossflow_data/recon_data/testset/'
    if save_dir is None:
        save_dir = '/storage/v-jinpewang/lab_folder/junchao/crossflow_data/test/recon_data/test_features'
    
    # 获取所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_files = []
    for filename in os.listdir(image_dir):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_files.append(filename)
    
    # 按文件名排序
    image_files.sort()
    
    # specify the path to the model
    model_path = "deepseek-ai/Janus-Pro-1B"
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, use_safetensors=True
    )
    vl_gpt = vl_gpt.half().to(device).eval()
    
    recreate_folder(save_dir)


    idx = 0
    batch_image_paths = []
    batch_names = []
    
    print("Total images:", len(image_files))
    print("Processing images in batches of {} - extracting features...".format(bz))
    
    # 禁用梯度计算，避免内存泄漏
    with torch.no_grad():
        for i, img_filename in enumerate(tqdm(image_files)):
            try:
                img_name = os.path.splitext(img_filename)[0]
                img_path = os.path.join(image_dir, img_filename)
                
                # 累积图片路径，准备批量处理
                batch_image_paths.append(img_path)
                batch_names.append(img_name)
                
            except Exception as e:
                with open("failed_file.txt", 'a+') as file: 
                    file.write(f"{img_filename}: {str(e)}\n")
                continue

            # 当累积到批次大小或到达最后一张图片时，进行批量推理
            if len(batch_names) == bz or i == len(image_files) - 1:
                # 批量加载图片
                question = ""
                all_pil_images = []
                for img_path in batch_image_paths:
                    try:
                        pil_img = Image.open(img_path).convert('RGB')
                        all_pil_images.append(pil_img)
                    except Exception as e:
                        print(f"Failed to load {img_path}: {e}")
                        all_pil_images.append(Image.new('RGB', (384, 384), color='black'))
                
                # 批量预处理
                images_outputs = vl_chat_processor.image_processor(all_pil_images, return_tensors="pt")
                batched_pixel_values = images_outputs.pixel_values.unsqueeze(1).to(device)  # [batch_size, 1, 3, H, W]
                
                # 准备批量文本输入
                sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
                    conversations=[
                        {"role": "<|User|>", "content": f"<image_placeholder>\n{question}"},
                        {"role": "<|Assistant|>", "content": ""},
                    ],
                    sft_format=vl_chat_processor.sft_format,
                    system_prompt="",
                )
                input_ids = vl_chat_processor.tokenizer.encode(sft_format)
                input_ids = torch.LongTensor(input_ids)
                
                # 准备批量输入
                batch_size = len(batch_names)
                batched_input_ids = input_ids.unsqueeze(0).repeat(batch_size, 1)  # [batch_size, seq_len]
                
                # 创建 image masks
                image_token_mask = batched_input_ids == vl_chat_processor.image_id
                batched_images_seq_mask = image_token_mask  # [batch_size, seq_len]
                
                # images_emb_mask 需要是 [batch_size, n_images, n_image_tokens]
                num_image_tokens = 576
                batched_images_emb_mask = torch.zeros((batch_size, 1, num_image_tokens)).bool()
                batched_images_emb_mask[:, :, :num_image_tokens] = True
                
                # 批量编码 - 一次处理所有图片
                inputs_embeds = vl_gpt.prepare_inputs_embeds(
                    input_ids=batched_input_ids.to(device),
                    pixel_values=batched_pixel_values,
                    images_seq_mask=batched_images_seq_mask.to(device),
                    images_emb_mask=batched_images_emb_mask.to(device)
                )  # [batch_size, 576, 2048]
                
                # 批量保存
                for batch_i, bn in enumerate(batch_names):
                    final_tensor = inputs_embeds[batch_i]  # [576, 2048]
                    attention_mask = [1] * 576
                    
                    assert final_tensor.shape == (576, 2048), f"Expected shape (576, 2048), got {final_tensor.shape}"
                    tar_path_name = os.path.join(save_dir, f'{bn}.npy')
                    if os.path.exists(tar_path_name):
                        os.remove(tar_path_name)
                    
                    # 验证集只保存必要的文本特征
                    data = {
                        'token_embedding_t5': final_tensor.detach().cpu().float().numpy(),
                        'token_mask_t5': attention_mask,
                    }
                    
                    try:
                        np.save(tar_path_name, data)
                        idx += 1
                    except Exception as e:
                        print(f'{idx} error: {e}')
                
                # 清空批次缓存
                batch_image_paths = []
                batch_names = []

    print(f'save {idx} files')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract test features using Janus model')
    parser.add_argument('--bz', '--batch_size', type=int, default=128, 
                        help='Batch size for processing (default: 128)')
    parser.add_argument('--device', type=str, default='cuda:7',
                        help='Device to use (default: cuda:7)')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Input image directory (default: /storage/.../testset/)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Output directory for features (default: /storage/.../test_features)')
    
    args = parser.parse_args()
    
    print(f"Configuration:")
    print(f"  Batch size: {args.bz}")
    print(f"  Device: {args.device}")
    print(f"  Image dir: {args.image_dir or 'default'}")
    print(f"  Save dir: {args.save_dir or 'default'}")
    print()
    
    main(bz=args.bz, device=args.device, image_dir=args.image_dir, save_dir=args.save_dir)