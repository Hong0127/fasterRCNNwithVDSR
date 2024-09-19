import torch
import torchvision.transforms.functional as F
from torchvision.transforms import RandomHorizontalFlip, ColorJitter

def sod4sb_transforms(image, target):
    # PIL Image를 PyTorch tensor로 변환
    image = F.to_tensor(image)
    
    # 랜덤 수평 뒤집기
    if torch.rand(1) < 0.5:
        image = F.hflip(image)
        bbox = target["boxes"]
        if bbox.shape[0] > 0:  # 박스가 있는 경우에만 변환
            bbox[:, [0, 2]] = image.shape[2] - bbox[:, [2, 0]]  # width를 기준으로 좌우 반전
        target["boxes"] = bbox

    # 밝기, 대비, 채도 조정
    color_jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
    image = color_jitter(image)

    return image, target

# 함수를 명시적으로 export
__all__ = ['sod4sb_transforms']