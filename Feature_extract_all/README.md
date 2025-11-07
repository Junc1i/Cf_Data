# feature extract
## environment
### 交互式环境下载
```sh
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install Cython
pip install -r requirements_cf.txt
pip install -r requirements_vae.txt
cd Janus
pip install -e .
pip install bitsandbytes
```
### job环境下载
```yaml
  setup: # 配置环境
    - pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
    - pip install -U deepspeed
    - pip install numpy==2.0.1
    - pip install pyarrow==21.0.0
    - pip install datasets==4.2.0
    - pip install scipy==1.15.3
    - pip install scikit-image==0.25.2
    - pip install scikit-learn==1.7.2 
    - pip install --user opencv-python==4.12.0.88 
    - pip install Cython
    - pip install openai-clip
    - pip install --no-deps torchdiffeq==0.2.5
    - pip install beautifulsoup4 
    - pip install open_clip_torch    
    - pip install cython 
    - pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI" 
    - pip install matplotlib
    - pip install accelerate==0.12.0    
    - pip install absl-py 
    - pip install ml-collections
    - pip install einops 
    - pip install wandb==0.22.2 
    - pip install ftfy==6.1.1 
    - pip install transformers==4.23.1 
    - pip install timm
    - pip install tensorboard
    - pip install pandas==2.3.3
    - cd /storage/v-jinpewang/lab_folder/qisheng_azure/Janus
    - pip install -e .
    - pip install -r requirements.txt
    - pip install -U "bitsandbytes>=0.48"
```
## image reconstrction task
### 1D model
先单独拿出30K样本（重建任务输入输出都是同一张图片，所以只要拿30k张图）作为testset，单独放一个文件夹。剩下的图片都作为训练样本。
#### 提取train feature
给模型的**输入输出都是同一张图片，指定一个trainset图片路径即可**
需要**修改sh中的相关配置**，下载image vae weights，记录下model weights路径（https://huggingface.co/turkeyju/tokenizer_tatitok_sl128_vae/tree/main）
使用**Crossflow/scripts/recon_task/run_multi_gpu.sh**运行八卡提取。运行后会保存目录下所有图片的npz文件

```sh
#!/bin/bash

# 批次大小（需要修改）
export BATCH_SIZE=128  # 可调整为512/2048，尽量往大开

# 需要修改
# 给模型的输入输出都是同一张图片，指定一个图片路径即可
export IMAGE_ROOT_PATH='/storage/v-jinpewang/lab_folder/junchao/crossflow_data/recon_data/trainset/train_01'
# 保存特征的路径
export SAVE_DIR='/storage/v-jinpewang/lab_folder/junchao/crossflow_data/test/recon_data/train_features_1D'

#（不需要修改）
export MODEL_PATH="deepseek-ai/Janus-Pro-1B"
# 指定image vae weights路径
export TATITOK_MODEL_PATH="/storage/v-jinpewang/lab_folder/junchao/Crossflow_1D/Img_VAE_Decoder/checkpoints/tatitok_bl128"

# GPU配置（需要修改）
export GPU_DEVICES=4,5 # 使用的GPU设备
export NUM_PROCESSES=2  # GPU数量

# DataLoader配置（不需要修改）
export NUM_WORKERS=8          # 数据加载器工作进程数
export PREFETCH_FACTOR=4      # 预取因子）
export RECURSIVE_SCAN=true    # 是否递归扫描子文件夹

# ==================== GPU和CUDA设置 ====================
# 设置可见的GPU（不需要修改）
export CUDA_VISIBLE_DEVICES=$GPU_DEVICES

# NCCL超时时间和优化设置（不需要修改）
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800  # 30分钟超时
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1      # 异步错误处理
export NCCL_TIMEOUT=1800                      # NCCL超时设置
export NCCL_DEBUG=WARN                        # 开启调试信息
export NCCL_IB_DISABLE=0                      # 使用InfiniBand
export NCCL_SOCKET_IFNAME=eth0                # 网络接口

# PyTorch CUDA 优化设置（不需要修改）
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True  # 减少内存碎片，启用可扩展段
export CUDA_LAUNCH_BLOCKING=0                 # 非阻塞模式
export OMP_NUM_THREADS=4                      # OpenMP线程数
export PYTORCH_ENABLE_MPS_FALLBACK=1          # 启用回退机制

# 使用accelerate启动脚本
# --num_processes: GPU数量
# --mixed_precision: 混合精度训练(可选: no, fp16, bf16)
# --multi_gpu: 启用多GPU模式

# 不需要修改
accelerate launch \
    --num_processes=$NUM_PROCESSES \
    --mixed_precision=fp16 \
    --multi_gpu \
    extract_train_feature.py 2>&1 | tee extract_recon_log.txt
    
run：
sh run_multi_gpu.sh
```
#### 提取test feature
使用**Crossflow/scripts/recon_task/extract_test_feature.py**运行单卡提取。
```python
bz: batch size
device: cuda
image_dir： testset图片路径
save_dir: 保存特征的路径

python extract_test_feature.py --bz 32 --device cuda:0 --image_dir "D:\test_images" --save_dir "D:\extracted_features"
```
#### 提取vis feature
从上面提取的test feature save_dir中取出15个npy文件单独放在一个路径。
### 2D model
先单独拿出30K样本（重建任务输入输出都是同一张图片，所以只要拿30k张图）作为testset，单独放一个文件夹。剩下的图片都作为训练样本。
#### 提取train feature
给模型的**输入输出都是同一张图片，指定一个trainset图片路径即可**
需要**修改sh中的相关配置**，下载image vae weights，记录下model weights路径（https://huggingface.co/QHL067/CrossFlow/blob/main/assets.tar）
使用**Crossflow/scripts/recon_task/run_multi_gpu_2D.sh**运行八卡提取。运行后会保存路径下所有图片的npz文件

```sh
#!/bin/bash
# 批次大小（需要修改）
export BATCH_SIZE=128  # 可调整为512/2048，尽量往大开

# 数据路径配置（需要修改）
# 给模型的输入输出都是同一张图片，指定一个图片路径即可
export IMAGE_ROOT_PATH='/storage/v-jinpewang/lab_folder/junchao/crossflow_data/recon_data/trainset/train_01'
# 保存特征的路径
export SAVE_DIR='/storage/v-jinpewang/lab_folder/junchao/crossflow_data/test/recon_data/train_features_2D'

# 模型路径配置
# （不需要修改）
export MODEL_PATH="deepseek-ai/Janus-Pro-1B"
# 需要修改为image vae weights路径
export AUTOENCODER_PATH="/storage/v-jinpewang/lab_folder/qisheng_data/assets/stable-diffusion/autoencoder_kl.pth"

# GPU配置（需要修改）
export GPU_DEVICES=6,7  # 使用的GPU设备
export NUM_PROCESSES=2    # GPU数量

# DataLoader配置（不需要修改）
export NUM_WORKERS=8          # 数据加载器工作进程数（减少以避免资源竞争）
export PREFETCH_FACTOR=4      # 预取因子（减少以降低内存压力）
export RECURSIVE_SCAN=true    # 是否递归扫描子文件夹

# ==================== GPU和CUDA设置 ====================
# 设置可见的GPU（不需要修改）
export CUDA_VISIBLE_DEVICES=$GPU_DEVICES

# NCCL超时时间和优化设置（不需要修改）
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800  # 30分钟超时
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1      # 异步错误处理
export NCCL_TIMEOUT=1800                      # NCCL超时设置
export NCCL_DEBUG=WARN                        # 开启调试信息
export NCCL_IB_DISABLE=0                      # 使用InfiniBand
export NCCL_SOCKET_IFNAME=eth0                # 网络接口

# PyTorch CUDA 优化设置（不需要修改）
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True  # 减少内存碎片，启用可扩展段
export CUDA_LAUNCH_BLOCKING=0                 # 非阻塞模式
export OMP_NUM_THREADS=4                      # OpenMP线程数
export PYTORCH_ENABLE_MPS_FALLBACK=1          # 启用回退机制

# 使用accelerate启动脚本
# --num_processes: GPU数量
# --mixed_precision: 混合精度训练(可选: no, fp16, bf16)
# --multi_gpu: 启用多GPU模式

#（不需要修改）
accelerate launch \
    --num_processes=$NUM_PROCESSES \
    --mixed_precision=fp16 \
    --multi_gpu \
    extract_train_feature_2D.py 2>&1 | tee extract_2D_log.txt

run:
sh run_multi_gpu_2D.sh
```
#### 提取test feature
使用**Crossflow/scripts/recon_task/extract_test_feature.py**运行单卡提取。
```python
bz: batch size
device: cuda
image_dir： testset图片路径
save_dir: 保存特征的路径

python extract_test_feature.py --bz 32 --device cuda:0 --image_dir "D:\test_images" --save_dir "D:\extracted_features"
```
#### 提取vis feature
从上面提取的test feature save_dir中取出15个npy文件单独放在一个路径。

## visual instruction task
### 1D model
先单独拿出60K样本（visual instruction任务输入输出都是不同图片，所以要从数据集路径下的input image dir拿30k张图,从output image dir拿30k张图）作为testset，单独放一个文件夹。剩下的图片都作为训练样本。
#### 提取train feature
给模型的输入输出不是同一张图片，**需要指定input和ouput image的路径，提取的token_embedding，toke_mask是input image的，z_mean,z_logvar是output image的**
需要**修改sh中的相关配置**，下载image vae weights，记录下model weights路径（https://huggingface.co/turkeyju/tokenizer_tatitok_sl128_vae/tree/main）
使用**Crossflow/scripts/visual_instuction_task/run_multi_gpu.sh**运行八卡提取。运行后会保存目录下所有图片的npz文件。

```sh
#!/bin/bash

# 批次大小（需要修改）
export BATCH_SIZE=128  # 可调整为512/2048，尽量往大开

# 数据路径配置（需要修改）
# 给模型的输入输出不是同一张图片，需要指定input,ouput image的目录
# 指定input image的路径
export INPUT_IMAGE_PATH='/storage/v-jinpewang/lab_folder/junchao/data/image_eidt_dataset/processed_data_wo_textbox/addtion/ultraedit/input/'  # 用于提取embeddings和masks
# 指定ouput image的路径
export OUTPUT_IMAGE_PATH='/storage/v-jinpewang/lab_folder/junchao/data/image_eidt_dataset/processed_data_wo_textbox/addtion/ultraedit/output/'  # 用于提取moments
# 保存特征的路径
export SAVE_DIR='/storage/v-jinpewang/lab_folder/junchao/crossflow_data/test/vis_data/train_features_1D'

# 模型路径配置
# 不需要修改
export MODEL_PATH="deepseek-ai/Janus-Pro-1B"
# 指定image vae weights路径
export TATITOK_MODEL_PATH="/storage/v-jinpewang/lab_folder/junchao/Crossflow_1D/Img_VAE_Decoder/checkpoints/tatitok_bl128"

# GPU配置（需要修改）
export GPU_DEVICES=4,5 # 使用的GPU设备
export NUM_PROCESSES=2  # GPU数量

# DataLoader配置（不需要需改）
export NUM_WORKERS=8          # 数据加载器工作进程数（减少以避免资源竞争）
export PREFETCH_FACTOR=4      # 预取因子（减少以降低内存压力）
export RECURSIVE_SCAN=true    # 是否递归扫描子文件夹

# ==================== GPU和CUDA设置 ====================
# 设置可见的GPU（不需要修改）
export CUDA_VISIBLE_DEVICES=$GPU_DEVICES

# NCCL超时时间和优化设置（不需要修改）
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800  # 30分钟超时
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1      # 异步错误处理
export NCCL_TIMEOUT=1800                      # NCCL超时设置
export NCCL_DEBUG=WARN                        # 开启调试信息
export NCCL_IB_DISABLE=0                      # 使用InfiniBand
export NCCL_SOCKET_IFNAME=eth0                # 网络接口

# PyTorch CUDA 优化设置（不需要修改）
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True  # 减少内存碎片，启用可扩展段
export CUDA_LAUNCH_BLOCKING=0                 # 非阻塞模式
export OMP_NUM_THREADS=4                      # OpenMP线程数
export PYTORCH_ENABLE_MPS_FALLBACK=1          # 启用回退机制

# 使用accelerate启动脚本
# --num_processes: GPU数量
# --mixed_precision: 混合精度训练(可选: no, fp16, bf16)
# --multi_gpu: 启用多GPU模式

# 不需要修改
accelerate launch \
    --num_processes=$NUM_PROCESSES \
    --mixed_precision=fp16 \
    --multi_gpu \
    extract_train_feature.py 2>&1 | tee extract_vis_log.txt
    
run：
sh run_multi_gpu.sh
```
#### 提取test feature

使用**Crossflow/scripts/visual_instuction_task/extract_test_feature.py**运行单卡提取。

```python
bz: batch size
device: cuda
image_dir： testset输入图片路径
save_dir: 保存特征的路径
python extract_test_feature.py --bz 32 --device cuda:0 --image_dir "D:\test_images" --save_dir "D:\extracted_features"
```

#### 提取vis feature

从上面提取的test feature save_dir中取出15个npy文件单独放在一个路径。

### 2D model

先单独拿出60K样本（visual instruction任务输入输出都是不同图片，所以要从input image dir拿30k张图,从output image dir拿30k张图）作为testset，单独放一个文件夹。剩下的图片都作为训练样本。

#### 提取train feature

给模型的输入输出不是同一张图片，**需要指定input和ouput image的路径，提取的token_embedding，toke_mask是input image的，z_mean,z_logvar是output image的**
需要**修改sh中的相关配置**，下载image vae weights，记录下model weights路径（https://huggingface.co/QHL067/CrossFlow/blob/main/assets.tar）
使用**Crossflow/scripts/visual_instuction_task/run_multi_gpu_2D.sh**运行八卡提取。运行后会保存目录下所有图片的npz文件。

```sh
#!/bin/bash

# 批次大小（需要修改）
export BATCH_SIZE=128  # 针对H100 80GB优化，可调整为512/2048，尽量往大开

# 数据路径配置（需要修改）
# 给模型的输入输出不是同一张图片，需要指定input,ouput image的目录
# 指定input image的路径
export INPUT_IMAGE_PATH='/storage/v-jinpewang/lab_folder/junchao/data/image_eidt_dataset/processed_data_wo_textbox/addtion/ultraedit/input/'  # 用于提取embeddings和masks
# 指定output image的路径
export OUTPUT_IMAGE_PATH='/storage/v-jinpewang/lab_folder/junchao/data/image_eidt_dataset/processed_data_wo_textbox/addtion/ultraedit/output/'  # 用于提取moments
# 指定保存特征的路径
export SAVE_DIR='/storage/v-jinpewang/lab_folder/junchao/crossflow_data/test/vis_data/train_features_2D'

# 模型路径配置
# 不需要修改
export MODEL_PATH="deepseek-ai/Janus-Pro-1B"
#（需要修改）指定image vae weights路径
export AUTOENCODER_PATH="/storage/v-jinpewang/lab_folder/qisheng_data/assets/stable-diffusion/autoencoder_kl.pth"

# GPU配置（需要修改）
export GPU_DEVICES=6,7  # 使用的GPU设备
export NUM_PROCESSES=2    # GPU数量

# DataLoader配置（不需要修改）
export NUM_WORKERS=8          # 数据加载器工作进程数（减少以避免资源竞争）
export PREFETCH_FACTOR=4      # 预取因子（减少以降低内存压力）
export RECURSIVE_SCAN=true    # 是否递归扫描子文件夹

# ==================== GPU和CUDA设置 ====================
# 设置可见的GPU（不需要修改）
export CUDA_VISIBLE_DEVICES=$GPU_DEVICES

# NCCL超时时间和优化设置（不需要修改）
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800  # 30分钟超时
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1      # 异步错误处理
export NCCL_TIMEOUT=1800                      # NCCL超时设置
export NCCL_DEBUG=WARN                        # 开启调试信息
export NCCL_IB_DISABLE=0                      # 使用InfiniBand
export NCCL_SOCKET_IFNAME=eth0                # 网络接口

# PyTorch CUDA 优化设置（不需要修改）
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True  # 减少内存碎片，启用可扩展段
export CUDA_LAUNCH_BLOCKING=0                 # 非阻塞模式
export OMP_NUM_THREADS=4                      # OpenMP线程数
export PYTORCH_ENABLE_MPS_FALLBACK=1          # 启用回退机制

# 使用accelerate启动脚本
# --num_processes: GPU数量
# --mixed_precision: 混合精度训练(可选: no, fp16, bf16)
# --multi_gpu: 启用多GPU模式

#（不需要修改）
accelerate launch \
    --num_processes=$NUM_PROCESSES \
    --mixed_precision=fp16 \
    --multi_gpu \
    extract_train_feature_2D.py 2>&1 | tee extract_vis_2D_log.txt

run:
sh run_multi_gpu_2D.sh
```

#### 提取test feature

使用**Crossflow/scripts/visual_instuction_task/extract_test_feature.py**运行单卡提取。

```python
bz: batch size
device: cuda
image_dir： testset输入图片路径
save_dir: 保存特征的路径

python extract_test_feature.py --bz 32 --device cuda:0 --image_dir "D:\test_images" --save_dir "D:\extracted_features"
```

#### 提取vis feature

从上面提取的test feature save_dir中取出15个npy文件单独放在一个路径。