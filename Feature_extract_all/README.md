# feature extract
## environment
```sh
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install Cython
pip install -r requirements_cf.txt
pip install -r requirements_vae.txt
cd Janus
pip install -e .
pip install bitsandbytes
```

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
## 1D model 
### task
#### recon task
使用Crossflow/scripts/run_multi_gpu.sh运行八卡提取,给模型的**输入输出都是同一张图片，指定一个目录即可**
需要**修改sh中的相关配置**，下载image vae weights（https://huggingface.co/turkeyju/tokenizer_tatitok_sl128_vae/tree/main）
运行后会保存目录下所有图片的npz文件

#### visual instruction task
给模型的输入输出不是同一张图片，**需要指定input,ouput image的目录，提取的token_embedding，toke_mask是input image的，z_mean,z_logvar是output image的**
### train feature
训练使用的特征，直接使用npz文件
### test feature
测试使用的特征，选取30k样本使用extract_test_feature.py处理。**只提取input image的特征**
### vis feature
训练可视化的特征，从测试集已经提取的特征中选取15个样本即可
## 2D model
#### recon task
使用Crossflow/scripts/extract_train_feature_2D.py运行八卡提取,**给模型的输入输出都是同一张图片，指定一个目录即可**
需要**修改sh中的相关配置**，下载image vae weights（https://huggingface.co/QHL067/CrossFlow/blob/main/assets.tar）
运行后会保存目录下所有图片的npz文件

#### visual instruction task
给模型的**输入输出不是同一张图片，需要指定input,ouput image的目录**，提取的token_embedding，toke_mask是input image的，z_mean,z_logvar是output image的
### train feature
训练使用的特征，直接使用npz文件
### test feature
测试使用的特征，选取30k样本使用Crossflow/scripts/extract_test_feature_2D.py处理。**只提取input image的特征**
### vis feature
训练可视化的特征，从测试集的样本中选取15张样本的特征单独放一个文件夹即可