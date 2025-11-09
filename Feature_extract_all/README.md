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
需要**修改sh中的相关配置**，下载image vae weights，记录下[model weights路径](https://huggingface.co/turkeyju/tokenizer_tatitok_sl128_vae/tree/main)
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
需要**修改sh中的相关配置**，下载image vae weights，记录下[model weights路径](https://huggingface.co/QHL067/CrossFlow/blob/main/assets.tar),使用assets/stable-diffusion/autoencoder_kl.pth
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
### 📃现有数据集说明
#### C2I dataset
简单input/output结构，按场景二提取训练集特征
```
根目录/
├── input/      # 输入图像（提取embeddings和masks）
│   │   │   ├── image1.jpg
│   │   │   ├── image2.png
│   │   │   └── ...
│── output/     # 输出图像（提取moments）
│   │       ├── image1.jpg
│   │       ├── image2.png
│   │       └── ...
```
#### T2I dataset
简单input/output结构，按场景二提取训练集特征
```
根目录/
├── input/      # 输入图像（提取embeddings和masks）
│   │   │   ├── image1.jpg
│   │   │   ├── image2.png
│   │   │   └── ...
│── output/     # 输出图像（提取moments）
│   │       ├── image1.jpg
│   │       ├── image2.png
│   │       └── ...
```
#### visual instruction dataset

[Junc1i/visual_instruction_dataset · Hugging Face](https://huggingface.co/Junc1i/visual_instruction_dataset)

复杂input/output结构，按场景一提取训练集特征
```
根目录/
├── addtion/
│   ├── omniedit/
│   │   ├── input/      # 输入图像（提取embeddings和masks）
│   │   │   ├── image1.jpg
│   │   │   ├── image2.png
│   │   │   └── ...
│   │   └── output/     # 输出图像（提取moments）
│   │       ├── image1.jpg
│   │       ├── image2.png
│   │       └── ...
│   └── ultraedit/
│       ├── input/
│       └── output/
├── attribute_modification/
│   ├── omniedit/
│   │   ├── input/
│   │   └── output/
│   └── ultraedit/
│       ├── input/
│       └── output/
└── ... (其他任务类型)
```
#### text_box精确生成
复杂input/output结构，按场景一提取训练集特征,但需要修改一下run_multi_gpu_batch.sh中的处理逻辑，可以直接把代码和以下目录丢给ai修改，只需要修改sh文件即可。
```
根目录/
├── with_textbox/
│   ├── input/      # 输入图像（提取embeddings和masks）
│   │   │   ├── image1.jpg
│   │   │   ├── image2.png
│   │   │   └── ...
│   └── output/     # 输出图像（提取moments）
│   │       ├── image1.jpg
│   │       ├── image2.png
│   │       └── ...
├── wo_textbox/
│   ├── input/
│   │   │   ├── image1.jpg
│   │   │   ├── image2.png
│   │   │   └── ...
│   ├── output/
│   │   │   ├── image1.jpg
│   │   │   ├── image2.png
│   │   │   └── ...
```
https://huggingface.co/datasets/Junc1i/Accgen_dataset

### 🚀 快速开始

**根据你的数据集类型和VAE选择对应的脚本，只针对训练集的特征提取。测试集的特征提取还是每个数据集任意拿出60K样本（visual instruction任务输入输出都是不同图片，所以要从数据集路径下的input image dir拿30k张图,从output image dir拿30k张图）作为testset，单独放一个文件夹运行extract_test_feature.py**

#### 场景1：提取Visual Instruction Dataset（多任务多类别）

**数据特点**：包含多个任务类型（addtion、attribute_modification等）和多类别（omniedit、ultraedit）

**使用脚本**：

```bash
# 1D VAE (TATiTok) - 批量处理所有任务
bash run_multi_gpu_batch.sh

# 2D VAE (Autoencoder) - 批量处理所有任务
bash run_multi_gpu_2D_batch.sh
```

**训练配置关键点**：
- `train_img_path` 必须指向 **ROOT_DIR**（包含所有任务的根目录）

---

#### 场景2：提取单一数据集（简单input/output结构）

**数据特点**：只有一对 input/output 文件夹（适用于C2I、T2I及text_box等数据集）

**使用脚本**：

```bash
# 1D VAE (TATiTok) - 单次处理
bash run_multi_gpu.sh

# 2D VAE (Autoencoder) - 单次处理
bash run_multi_gpu_2D.sh
```

**训练配置关键点**：
- `train_img_path` 必须指向 **OUTPUT_IMAGE_PATH**（单个output目录）

---

### 📁 目录结构要求

Visual Instruction Dataset为以下结构组织：

```
根目录/
├── addtion/
│   ├── omniedit/
│   │   ├── input/      # 输入图像（提取embeddings和masks）
│   │   │   ├── image1.jpg
│   │   │   ├── image2.png
│   │   │   └── ...
│   │   └── output/     # 输出图像（提取moments）
│   │       ├── image1.jpg
│   │       ├── image2.png
│   │       └── ...
│   └── ultraedit/
│       ├── input/
│       └── output/
├── attribute_modification/
│   ├── omniedit/
│   │   ├── input/
│   │   └── output/
│   └── ultraedit/
│       ├── input/
│       └── output/
└── ... (其他任务类型)
```

---

### 🚀 处理脚本对比

| 脚本 | VAE类型 | 适用场景 | 数据集类型 | 特点 |
|------|---------|---------|-----------|------|
| `run_multi_gpu.sh` | 1D (TATiTok) | 处理单个input/output对 | 通用 | 原始版本，单次处理 |
| `run_multi_gpu_2D.sh` | 2D (Autoencoder) | 处理单个input/output对 | 通用 | 原始版本，单次处理 |
| `run_multi_gpu_batch.sh` | 1D (TATiTok) | 专用于Visual Instruction数据集 | Visual Instruction | 全自动，统一保存，支持多任务 |
| `run_multi_gpu_2D_batch.sh` | 2D (Autoencoder) | 专用于Visual Instruction数据集 | Visual Instruction | 全自动，统一保存，支持多任务 |

#### ⚠️ 重要说明

**批处理脚本专用于 Visual Instruction Dataset**：

- `run_multi_gpu_batch.sh` 和 `run_multi_gpu_2D_batch.sh` 专为 Visual Instruction 数据集设计
- 这类数据集通常包含多个任务类型（如 addtion、attribute_modification 等）和多种类别（omniedit、ultraedit）
- 特点是需要从不同的 input/output 文件夹对中提取特征，并统一保存以支持混合训练

**单任务脚本适用于其他数据集**：
- `run_multi_gpu.sh` 和 `run_multi_gpu_2D.sh` 适合单一数据集，C2I，T2I以及后面的text_box等数据集

---

### 方案1: 单任务处理（适用于C2I、T2I等数据集）

先单独拿出60K样本（visual instruction任务输入输出都是不同图片，所以要从数据集路径下的input image dir拿30k张图,从output image dir拿30k张图）作为testset，单独放一个文件夹。剩下的图片都作为训练样本。

#### 1D Model - 单任务处理

##### 提取train feature

给模型的输入输出不是同一张图片，**需要指定input和ouput image的路径，提取的token_embedding，toke_mask是input image的，z_mean,z_logvar是output image的**

需要**修改sh中的相关配置**，下载image vae weights，记录下[model weights路径](https://huggingface.co/turkeyju/tokenizer_tatitok_sl128_vae/tree/main)

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

##### 提取test feature

使用**Crossflow/scripts/visual_instuction_task/extract_test_feature.py**运行单卡提取。

```python
bz: batch size
device: cuda
image_dir： testset输入图片路径
save_dir: 保存特征的路径
python extract_test_feature.py --bz 32 --device cuda:0 --image_dir "D:\test_images" --save_dir "D:\extracted_features"
```

##### 提取vis feature

从上面提取的test feature save_dir中取出15个npy文件单独放在一个路径。

---

#### 2D Model - 单任务处理

##### 提取train feature

给模型的输入输出不是同一张图片，**需要指定input和ouput image的路径，提取的token_embedding，toke_mask是input image的，z_mean,z_logvar是output image的**

需要**修改sh中的相关配置**，下载image vae weights，记录下[model weights路径](https://huggingface.co/QHL067/CrossFlow/blob/main/assets.tar),使用assets/stable-diffusion/autoencoder_kl.pth

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
# 下面的都不需要修改
```

##### 提取test feature

使用**Crossflow/scripts/visual_instuction_task/extract_test_feature.py**运行单卡提取。

```python
bz: batch size
device: cuda
image_dir： testset输入图片路径
save_dir: 保存特征的路径

python extract_test_feature.py --bz 32 --device cuda:0 --image_dir "D:\test_images" --save_dir "D:\extracted_features"
```

##### 提取vis feature

从上面提取的test feature save_dir中取出15个npy文件单独放在一个路径。

---

### 方案2: 批量处理（专用于Visual Instruction Dataset）

> **🎯 专用于包含多任务类型和多类别的 Visual Instruction Dataset**

#### 1D版本：run_multi_gpu_batch.sh (TATiTok)

##### 适用场景
使用TATiTok进行1D VAE编码，一次性处理所有任务类型和编辑方法

##### 使用方法

**步骤1: 修改配置**

编辑 `run_multi_gpu_batch.sh`：

```bash
# 根目录配置（需要修改）
ROOT_DIR='/storage/v-jinpewang/lab_folder/junchao/data/image_eidt_dataset/processed_data_wo_textbox'
SAVE_ROOT_DIR='/storage/v-jinpewang/lab_folder/junchao/crossflow_data/test/vis_data/train_features_1D'

# 批次大小（需要修改）
export BATCH_SIZE=128  # 针对H100 80GB优化

# 模型路径配置
# （不需要修改）
export MODEL_PATH="deepseek-ai/Janus-Pro-1B"
#（需要修改）
export TATITOK_MODEL_PATH="/storage/v-jinpewang/lab_folder/junchao/Crossflow_1D/Img_VAE_Decoder/checkpoints/tatitok_bl128"

# GPU配置（需要修改）
export GPU_DEVICES=4,5  # 使用的GPU设备
export NUM_PROCESSES=2  # GPU数量

# DataLoader配置（不需要修改）
export NUM_WORKERS=8          # 数据加载器工作进程数
export PREFETCH_FACTOR=4      # 预取因子
export RECURSIVE_SCAN=true    # 递归扫描子文件夹

# ==================== GPU和CUDA设置 ====================
#（不需要修改）
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
# 下面的都不需要修改
```

**步骤2: 运行脚本**

```bash
chmod +x run_multi_gpu_batch.sh
bash run_multi_gpu_batch.sh
```

**输出文件名格式**：
- NPZ文件：`batch_{task_type}__{edit_method}_{run_id}_{batch_idx}_rank{rank}.npz`
- 日志文件：`extract_{task_type}_{edit_method}_{timestamp}.log`

---

#### 2D版本：run_multi_gpu_2D_batch.sh (Autoencoder)

##### 适用场景
使用Autoencoder进行2D VAE编码，一次性处理所有任务类型和编辑方法

##### 使用方法

**步骤1: 修改配置**

编辑 `run_multi_gpu_2D_batch.sh`：

```bash
# 根目录配置（需要修改）
ROOT_DIR='/storage/v-jinpewang/lab_folder/junchao/data/image_eidt_dataset/processed_data_wo_textbox'
SAVE_ROOT_DIR='/storage/v-jinpewang/lab_folder/junchao/crossflow_data/test/vis_data/train_features_2D'

# 批次大小（需要修改）
export BATCH_SIZE=128  # 针对H100 80GB优化

# 模型路径配置（需要修改）
# （不需要修改）
export MODEL_PATH="deepseek-ai/Janus-Pro-1B"
export AUTOENCODER_PATH="/storage/v-jinpewang/lab_folder/qisheng_data/assets/stable-diffusion/autoencoder_kl.pth"

# GPU配置（需要修改）
export GPU_DEVICES=6,7  # 使用的GPU设备
export NUM_PROCESSES=2  # GPU数量

# DataLoader配置（需要修改）
export NUM_WORKERS=8          # 数据加载器工作进程数
export PREFETCH_FACTOR=4      # 预取因子
export RECURSIVE_SCAN=true    # 递归扫描子文件夹

# ==================== GPU和CUDA设置 ====================
（不需要修改）
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
```

**步骤2: 运行脚本**

```bash
bash run_multi_gpu_2D_batch.sh
```

**输出文件名格式**：
- NPZ文件：`batch_{task_type}__{edit_method}_{run_id}_{batch_idx}_rank{rank}.npz`
- 日志文件：`extract_{task_type}_{edit_method}_{timestamp}.log`

##### 输出结构

```
SAVE_ROOT_DIR/
├── batch_addtion__omniedit_20241109_143022_000001_rank0.npz
├── batch_addtion__omniedit_20241109_143022_000002_rank1.npz
├── batch_addtion__ultraedit_20241109_145030_000001_rank0.npz
├── batch_addtion__ultraedit_20241109_145030_000002_rank1.npz
├── batch_attribute_modification__omniedit_20241109_150100_000001_rank0.npz
├── batch_attribute_modification__omniedit_20241109_150100_000002_rank1.npz
├── extract_addtion_omniedit_20241109_143022.log
├── extract_addtion_ultraedit_20241109_145030.log
├── extract_attribute_modification_omniedit_20241109_150100.log
├── processing_log.txt  # 总体处理日志
└── batch_extract_vis_2D_20241109_143000.log  # 总日志
```

**说明**：
- 所有NPZ文件统一保存在 SAVE_ROOT_DIR 中
- 文件名包含任务标识：`batch_{task_type}__{edit_method}_{run_id}_{batch_idx}_rank{rank}.npz`
- 双下划线 `__` 分隔任务类型和编辑方法
- 每个任务组合有独立的处理日志

##### 两个batch版本的比较

| 特性 | 1D版本 (TATiTok) | 2D版本 (Autoencoder) |
|------|------------------|---------------------|
| VAE输出 | means + logvars | moments |
| 图像归一化 | [0, 1] | [-1, 1] |
| 输出形状 | [16, 1, 128] | [8, 32, 32] |
| 模型参数 | TATITOK_MODEL_PATH | AUTOENCODER_PATH |

---

### 🔧 重要配置参数

#### GPU配置

```bash
export GPU_DEVICES=6,7     # 使用的GPU编号
export NUM_PROCESSES=2     # GPU数量（必须与GPU_DEVICES数量一致）
```

#### 批次大小

```bash
export BATCH_SIZE=128      # 根据GPU显存调整
                           # H100 80GB: 可用 512-2048
                           # V100 32GB: 建议 128-256
```

#### DataLoader配置

```bash
export NUM_WORKERS=8       # 数据加载线程数
export PREFETCH_FACTOR=4   # 预取因子
export RECURSIVE_SCAN=true # 是否递归扫描子文件夹
```

---

### 📊 输出说明

#### NPZ文件内容

##### 批处理脚本生成的NPZ文件（专用于visual instruction dataset）

每个 `.npz` 文件包含：
- `sample_names`: 样本名称（数字键）
- `input_image_relative_paths`: 输入图像相对路径（**相对于ROOT_DIR**）
- `output_image_relative_paths`: 输出图像相对路径（**相对于ROOT_DIR**）
- `embeddings`: Janus提取的token embeddings [batch_size, 576, 2048]
- `masks`: attention masks [batch_size, 576]
- `moments`: Autoencoder提取的moments [batch_size, 8, 32, 32] (2D)
- `means` + `logvars`: TATiTok提取的分布参数 [batch_size, 16, 1, 128] (1D)
- `vae_type`: '2D' 或 '1D'
- `llm`: 't5'
- `resolution`: 256
- `task_type`: 任务类型（如 'addtion', 'attribute_modification' 等）
- `edit_method`: 多类别（如 'omniedit', 'ultraedit' 等）

##### 单任务脚本生成的NPZ文件

每个 `.npz` 文件包含：
- `sample_names`: 样本名称（数字键）
- `input_image_relative_paths`: 输入图像相对路径（**相对于INPUT_IMAGE_PATH**）
- `output_image_relative_paths`: 输出图像相对路径（**相对于OUTPUT_IMAGE_PATH**）
- `embeddings`: Janus提取的token embeddings [batch_size, 576, 2048]
- `masks`: attention masks [batch_size, 576]
- `moments`: Autoencoder提取的moments [batch_size, 8, 32, 32] (2D)
- `means` + `logvars`: TATiTok提取的分布参数 [batch_size, 16, 1, 128] (1D)
- `vae_type`: '2D' 或 '1D'
- `llm`: 't5'
- `resolution`: 256
- ❌ **没有** `task_type` 和 `edit_method` 字段

#### 相对路径说明

**处理visual instruction dataset的脚本（支持多任务混合训练）**：

假设目录结构：

```
ROOT_DIR/
├── addtion/
│   └── ultraedit/
│       ├── input/
│       │   └── img1.jpg
│       └── output/
│           └── img1.jpg
└── attribute_modification/
    └── omniedit/
        ├── input/
        │   └── subfolder/
        │       └── img2.jpg
        └── output/
            └── subfolder/
                └── img2.jpg
```

NPZ文件中保存的相对路径（相对于ROOT_DIR）：
- `input_image_relative_paths`: 
  - `'addtion/ultraedit/input/img1.jpg'`
  - `'attribute_modification/omniedit/input/subfolder/img2.jpg'`
- `output_image_relative_paths`:
  - `'addtion/ultraedit/output/img1.jpg'`
  - `'attribute_modification/omniedit/output/subfolder/img2.jpg'`

---

### ⚠️ 重要：训练配置必须与特征提取脚本匹配

训练配置必须与特征提取时使用的脚本相对应，否则会导致图像路径错误！

#### 配置1：使用批处理脚本提取的特征（专用于Visual Instruction Dataset）

**适用于**：使用 `run_multi_gpu_batch.sh` 或 `run_multi_gpu_2D_batch.sh` 提取的特征

```python
# configs/t2i_training_visual_instruction.py

config.dataset = d(
    name='textimage_features',
    resolution=256,
    llm='t5',
    
    # ✅ 批处理脚本：train_feature_dir 指向统一保存的 SAVE_ROOT_DIR
    train_feature_dir='/storage/.../train_features_2D',
    
    # ✅ 批处理脚本：train_img_path 必须指向包含所有任务的根目录 (ROOT_DIR)
    # 因为NPZ中保存的路径格式是：'addtion/ultraedit/output/img.jpg'
    train_img_path='/storage/.../processed_data_wo_textbox',  # ROOT_DIR
    
    val_feature_dir='/storage/.../val_features',
    run_vis_feature_dir='/storage/.../run_vis',
    cfg=False
)
```

**关键点**：
- ✅ NPZ文件包含完整路径：`'task_type/edit_method/output/relative_path'`
- ✅ `train_img_path` 指向 **ROOT_DIR**（包含所有任务类型的根目录）
- ✅ 最终路径 = `ROOT_DIR` + `task_type/edit_method/output/img.jpg`
- ✅ 支持混合训练多个任务和编辑方法

---

#### 配置2：使用单任务脚本提取的特征

**适用于**：使用 `run_multi_gpu.sh` 或 `run_multi_gpu_2D.sh` 提取的特征

```python
# configs/t2i_training_single_task.py

config.dataset = d(
    name='textimage_features',
    resolution=256,
    llm='t5',
    
    # ✅ 单任务脚本：train_feature_dir 指向特征保存目录
    train_feature_dir='/storage/.../train_features_single',
    
    # ✅ 单任务脚本：train_img_path 必须指向单个任务的 output 目录
    # 因为NPZ中保存的路径格式是：'img.jpg'
    train_img_path='/storage/.../addtion/ultraedit/output',  # OUTPUT_IMAGE_PATH
    
    val_feature_dir='/storage/.../val_features',
    run_vis_feature_dir='/storage/.../run_vis',
    cfg=False
)
```

**关键点**：
- ✅ NPZ文件只包含相对路径：`'img.jpg'` 或 `'subfolder/img.jpg'`
- ✅ `train_img_path` 指向 **单个任务的output目录**
- ✅ 最终路径 = `OUTPUT_IMAGE_PATH` + `img.jpg`
- ✅ 只能训练单个任务

---

#### ❌ 常见错误配置

**错误1：批处理特征 + 单任务路径**

```python
# 使用 run_multi_gpu_2D_batch.sh 提取的特征
train_feature_dir='/storage/.../train_features_2D'
train_img_path='/storage/.../addtion/ultraedit/output'  # ❌ 错误！

# 问题：NPZ中路径是 'addtion/ultraedit/output/img.jpg'
# 拼接后：'/storage/.../addtion/ultraedit/output/addtion/ultraedit/output/img.jpg'
# 结果：FileNotFoundError（路径重复）
```

**错误2：单任务特征 + 批处理路径**

```python
# 使用 run_multi_gpu_2D.sh 提取的特征
train_feature_dir='/storage/.../train_features_single'
train_img_path='/storage/.../processed_data_wo_textbox'  # ❌ 错误！

# 问题：NPZ中路径是 'img.jpg'
# 拼接后：'/storage/.../processed_data_wo_textbox/img.jpg'
# 结果：FileNotFoundError（缺少中间目录结构）
```

---

### 📋 配置检查清单

训练前请确认：

- [ ] 确认使用的是哪个脚本提取的特征（批处理 or 单任务）
- [ ] 检查NPZ文件中的路径格式
- [ ] 根据路径格式设置正确的 `train_img_path`

---

### 📝 日志文件说明

#### 总日志 (batch_extract_vis_2D_*.log)
记录所有任务的成功/失败状态

#### 任务日志 ({SAVE_DIR}/extract_log_*.txt)
每个任务的详细处理日志

#### 处理日志 ({SAVE_DIR}/processing_log.txt)
记录详细的处理统计信息

---

### ⚡ 建议

#### 对于H100 80GB

```bash
export BATCH_SIZE=512        # 或更大
export NUM_WORKERS=16
export PREFETCH_FACTOR=8
```

#### 对于V100 32GB

```bash
export BATCH_SIZE=128
export NUM_WORKERS=8
export PREFETCH_FACTOR=4
```