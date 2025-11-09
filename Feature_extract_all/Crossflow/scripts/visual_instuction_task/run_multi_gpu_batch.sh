#!/bin/bash

# 批量多GPU并行处理脚本 - 1D特征提取版本（TATiTok）
# 🎯 专用于 Visual Instruction Dataset
# 自动遍历所有任务类型和编辑方法，提取特征并统一保存
# 
# 特点：
# - 自动检测 ROOT_DIR 下的所有任务类型和编辑方法
# - 统一保存到 SAVE_ROOT_DIR，支持混合训练
# - NPZ文件包含完整的相对路径（相对于ROOT_DIR）

# ==================== 重要变量配置 ====================
# 根目录配置
ROOT_DIR='/storage/v-jinpewang/lab_folder/junchao/data/image_eidt_dataset/processed_data_wo_textbox'
SAVE_ROOT_DIR='/storage/v-jinpewang/lab_folder/junchao/crossflow_data/test/vis_data/train_features_1D'

# 批次大小（根据GPU显存调整）
export BATCH_SIZE=128  # 针对H100 80GB优化

# 模型路径配置
export MODEL_PATH="deepseek-ai/Janus-Pro-1B"
export TATITOK_MODEL_PATH="/storage/v-jinpewang/lab_folder/junchao/Crossflow_1D/Img_VAE_Decoder/checkpoints/tatitok_bl128"

# GPU配置
export GPU_DEVICES=4,5  # 使用的GPU设备
export NUM_PROCESSES=2  # GPU数量

# DataLoader配置
export NUM_WORKERS=8          # 数据加载器工作进程数
export PREFETCH_FACTOR=4      # 预取因子
export RECURSIVE_SCAN=true    # 递归扫描子文件夹

# ==================== GPU和CUDA设置 ====================
export CUDA_VISIBLE_DEVICES=$GPU_DEVICES
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=4
export PYTORCH_ENABLE_MPS_FALLBACK=1

# ==================== 任务类型和编辑方法列表 ====================
# 定义所有任务类型（对应文件夹名）
TASK_TYPES=(
    "addtion"
    "attribute_modification"
    "background_swap"
    "change_color"
    "change_global"
    "change_local"
    "env"
    "object_swap"
    "others"
    "removal"
    "replace"
    "style"
    "swap"
    "transform_global"
    "transform_local"
    "turn"
)

# 可能的编辑方法（脚本会自动检测实际存在的）
POSSIBLE_EDIT_METHODS=(
    "omniedit"
    "ultraedit"
    # 如果还有其他编辑方法，在这里添加
)

# ==================== 批量处理 ====================
echo "=========================================="
echo "开始批量特征提取（1D - TATiTok）"
echo "根目录: $ROOT_DIR"
echo "保存根目录: $SAVE_ROOT_DIR"
echo "任务类型数量: ${#TASK_TYPES[@]}"
echo "=========================================="
echo ""

# 创建保存目录
mkdir -p "$SAVE_ROOT_DIR"

# 统计信息
current_index=0
processed_count=0
skipped_count=0
failed_count=0

# 创建总日志文件
MAIN_LOG_FILE="$SAVE_ROOT_DIR/batch_extract_vis_1D_$(date +%Y%m%d_%H%M%S).log"
echo "总日志文件: $MAIN_LOG_FILE"
echo ""

# 先统计总组合数
total_combinations=0
for task_type in "${TASK_TYPES[@]}"; do
    task_dir="$ROOT_DIR/$task_type"
    if [ ! -d "$task_dir" ]; then
        continue
    fi
    
    for edit_method in "${POSSIBLE_EDIT_METHODS[@]}"; do
        INPUT_PATH="$task_dir/$edit_method/input"
        OUTPUT_PATH="$task_dir/$edit_method/output"
        
        if [ -d "$INPUT_PATH" ] && [ -d "$OUTPUT_PATH" ]; then
            total_combinations=$((total_combinations + 1))
        fi
    done
done

echo "检测到 $total_combinations 个有效的任务组合"
echo ""

# 遍历所有任务类型
for task_type in "${TASK_TYPES[@]}"; do
    task_dir="$ROOT_DIR/$task_type"
    
    # 检查任务类型文件夹是否存在
    if [ ! -d "$task_dir" ]; then
        echo "⚠️  任务类型文件夹不存在，跳过: $task_type"
        continue
    fi
    
    # 自动检测该任务类型下存在的编辑方法
    for edit_method in "${POSSIBLE_EDIT_METHODS[@]}"; do
        # 构建路径
        INPUT_PATH="$task_dir/$edit_method/input"
        OUTPUT_PATH="$task_dir/$edit_method/output"
        
        # 检查路径是否存在（先检查，再计数）
        if [ ! -d "$INPUT_PATH" ]; then
            continue  # 静默跳过，不显示信息
        fi
        
        if [ ! -d "$OUTPUT_PATH" ]; then
            continue  # 静默跳过，不显示信息
        fi
        
        # 只有路径存在时才递增计数器
        current_index=$((current_index + 1))
        
        echo "=========================================="
        echo "[$current_index/$total_combinations] 处理: $task_type / $edit_method"
        echo "=========================================="
        echo "输入路径: $INPUT_PATH"
        echo "输出路径: $OUTPUT_PATH"
        echo "保存路径: $SAVE_ROOT_DIR (统一存储)"
        
        # 检查input和output文件夹是否有图片
        input_count=$(find "$INPUT_PATH" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" -o -iname "*.webp" \) 2>/dev/null | wc -l)
        output_count=$(find "$OUTPUT_PATH" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" -o -iname "*.webp" \) 2>/dev/null | wc -l)
        
        echo "图片数量: input=$input_count, output=$output_count"
        
        if [ $input_count -eq 0 ] || [ $output_count -eq 0 ]; then
            echo "⚠️  没有图片文件，跳过"
            skipped_count=$((skipped_count + 1))
            echo ""
            continue
        fi
        
        # 设置环境变量
        export INPUT_IMAGE_PATH="$INPUT_PATH"
        export OUTPUT_IMAGE_PATH="$OUTPUT_PATH"
        export SAVE_DIR="$SAVE_ROOT_DIR"
        
        # 设置任务标识（用于NPZ文件名前缀）
        export TASK_PREFIX="${task_type}__${edit_method}"
        
        # 创建单独的日志文件（放在SAVE_ROOT_DIR）
        LOG_FILE="$SAVE_ROOT_DIR/extract_${task_type}_${edit_method}_$(date +%Y%m%d_%H%M%S).log"
        
        echo "任务标识: $TASK_PREFIX"
        echo "VAE类型: 1D (TATiTok)"
        echo "日志文件: $(basename $LOG_FILE)"
        echo ""
        echo "✓ 开始处理..."
        echo ""
        
        # 运行特征提取（1D版本）
        accelerate launch \
            --num_processes=$NUM_PROCESSES \
            --mixed_precision=fp16 \
            --multi_gpu \
            extract_train_feature.py 2>&1 | tee "$LOG_FILE"
        
        # 检查退出状态
        exit_status=${PIPESTATUS[0]}
        echo ""
        
        if [ $exit_status -eq 0 ]; then
            echo "✓ 处理完成: $task_type / $edit_method"
            processed_count=$((processed_count + 1))
            
            # 追加到总日志
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS: $task_type/$edit_method (input=$input_count, output=$output_count)" >> "$MAIN_LOG_FILE"
        else
            echo "✗ 处理失败: $task_type / $edit_method (退出码: $exit_status)"
            failed_count=$((failed_count + 1))
            
            # 追加到总日志
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED: $task_type/$edit_method (exit_code=$exit_status)" >> "$MAIN_LOG_FILE"
        fi
        
        echo ""
        
        # 任务间短暂延迟，避免资源竞争
        sleep 2
    done
done

# ==================== 最终统计 ====================
echo ""
echo "=========================================="
echo "批量处理完成！"
echo "=========================================="
echo "总组合数: $total_combinations"
echo "✓ 成功处理: $processed_count"
echo "⚠️  跳过: $skipped_count"
echo "✗ 失败: $failed_count"
echo ""
echo "所有特征文件保存在: $SAVE_ROOT_DIR"
echo "详细日志: $(basename $MAIN_LOG_FILE)"
echo "=========================================="

# 显示成功和失败的详细列表
if [ $processed_count -gt 0 ]; then
    echo ""
    echo "成功处理的任务:"
    grep "SUCCESS" "$MAIN_LOG_FILE" 2>/dev/null | sed 's/^/  /' || echo "  (无)"
fi

if [ $failed_count -gt 0 ]; then
    echo ""
    echo "失败的任务:"
    grep "FAILED" "$MAIN_LOG_FILE" 2>/dev/null | sed 's/^/  /' || echo "  (无)"
fi

echo ""

# 退出码：如果有失败的任务，返回1
if [ $failed_count -gt 0 ]; then
    exit 1
else
    exit 0
fi

