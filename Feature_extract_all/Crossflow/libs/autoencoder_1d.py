import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

# TA-TiTok相关导入
try:
    from .vae1d.modeling.tatitok import TATiTok
    from .vae1d.modeling.quantizer.quantizer import DiagonalGaussianDistribution
except ImportError:
    # 如果导入失败,提供友好的错误信息
    print("无法导入TA-TiTok相关模块,请确保它们在Python路径中可用")
    TATiTok = None
    DiagonalGaussianDistribution = None


def get_model(pretrained_path):
    """
    加载TA-TiTok模型
    
    Args:
        pretrained_path: TA-TiTok模型的路径或HuggingFace模型名称
    
    Returns:
        TATiTok: 加载的TA-TiTok模型
    """
    if TATiTok is None:
        raise ImportError("TA-TiTok模块未正确导入")
    
    print(f'Loading TA-TiTok model from {pretrained_path}')
    model = TATiTok.from_pretrained(pretrained_path)
    model.eval()
    model.requires_grad_(False)
    return model


# def main():
#     import torchvision.transforms as transforms
#     from torchvision.utils import save_image
#     import os
#     from PIL import Image

#     # 使用TA-TiTok模型
#     model = get_model('bytedance/TATiTok')  # 或者使用本地路径
#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#     model = model.to(device)

#     T = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor()])
#     path = 'imgs'
#     fnames = os.listdir(path)
#     for fname in fnames:
#         p = os.path.join(path, fname)
#         img = Image.open(p)
#         img = T(img)
#         img = img * 2. - 1
#         img = img[None, ...]
#         img = img.to(device)

#         with torch.cuda.amp.autocast():
#             print('test encode & decode with TA-TiTok')
#             # 直接使用TA-TiTok的encode和decode方法
#             z_quantized, result_dict = model.encode(img)
#             recons = [model.decode(z_quantized, text_guidance=None) for _ in range(4)]

#         out = torch.cat([img, *recons], dim=0)
#         out = (out + 1) * 0.5
#         save_image(out, f'recons_tatitok_{fname}')


# if __name__ == "__main__":
#     main()

