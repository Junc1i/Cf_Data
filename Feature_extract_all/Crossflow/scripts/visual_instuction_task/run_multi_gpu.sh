#!/bin/bash

# 多GPU并行处理脚本 - 1D特征提取版本（TATiTok）
# 适用于单一数据集或单个任务
# 使用accelerate launch启动多卡并行
# 
# 特点：
# - 处理单个 input/output 文件夹对
# - NPZ文件包含相对路径（相对于INPUT_IMAGE_PATH和OUTPUT_IMAGE_PATH）
# - 训练时需要设置 train_img_path=OUTPUT_IMAGE_PATH

# ==================== 重要变量配置 ====================
# 批次大小（根据GPU显存调整）
export BATCH_SIZE=128  # 针对H100 80GB优化，可调整为512/2048，尽量往大开

# 数据路径配置
export INPUT_IMAGE_PATH='/storage/v-jinpewang/lab_folder/junchao/data/image_eidt_dataset/processed_data_wo_textbox/addtion/ultraedit/input/'  # 用于提取embeddings和masks
export OUTPUT_IMAGE_PATH='/storage/v-jinpewang/lab_folder/junchao/data/image_eidt_dataset/processed_data_wo_textbox/addtion/ultraedit/output/'  # 用于提取moments
export SAVE_DIR='/storage/v-jinpewang/lab_folder/junchao/crossflow_data/test/vis_data/train_features_1D'

# 模型路径配置
export MODEL_PATH="deepseek-ai/Janus-Pro-1B"
export TATITOK_MODEL_PATH="/storage/v-jinpewang/lab_folder/junchao/Crossflow_1D/Img_VAE_Decoder/checkpoints/tatitok_bl128"

# GPU配置
export GPU_DEVICES=4,5 # 使用的GPU设备
export NUM_PROCESSES=2  # GPU数量

# DataLoader配置
export NUM_WORKERS=8          # 数据加载器工作进程数（减少以避免资源竞争）
export PREFETCH_FACTOR=4      # 预取因子（减少以降低内存压力）
export RECURSIVE_SCAN=true    # 是否递归扫描子文件夹

# ==================== GPU和CUDA设置 ====================
# 设置可见的GPU（根据需要修改）
export CUDA_VISIBLE_DEVICES=$GPU_DEVICES

# NCCL超时时间和优化设置
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800  # 30分钟超时
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1      # 异步错误处理
export NCCL_TIMEOUT=1800                      # NCCL超时设置
export NCCL_DEBUG=WARN                        # 开启调试信息
export NCCL_IB_DISABLE=0                      # 使用InfiniBand
export NCCL_SOCKET_IFNAME=eth0                # 网络接口

# PyTorch CUDA 优化设置
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True  # 减少内存碎片，启用可扩展段
export CUDA_LAUNCH_BLOCKING=0                 # 非阻塞模式
export OMP_NUM_THREADS=4                      # OpenMP线程数
export PYTORCH_ENABLE_MPS_FALLBACK=1          # 启用回退机制

# 使用accelerate启动脚本
# --num_processes: GPU数量
# --mixed_precision: 混合精度训练(可选: no, fp16, bf16)
# --multi_gpu: 启用多GPU模式

accelerate launch \
    --num_processes=$NUM_PROCESSES \
    --mixed_precision=fp16 \
    --multi_gpu \
    extract_train_feature.py 2>&1 | tee extract_vis_log.txt
