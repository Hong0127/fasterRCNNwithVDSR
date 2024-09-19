import os
import torch
import torch.nn as nn
from math import sqrt

class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.conv(x))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 18)
        self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out,residual)
        return out

def load_vdsr_weights(model, weights_path):
    if os.path.isfile(weights_path):
        print(f"=> loading VDSR model from '{weights_path}'")
        weights = torch.load(weights_path, map_location=torch.device('cpu'))
        
        if 'model' in weights:
            # 원본 코드 방식대로 'model' 키를 통해 상태 딕셔너리에 접근
            state_dict = weights['model'].state_dict()
        else:
            # 'model' 키가 없는 경우 직접 상태 딕셔너리로 간주
            state_dict = weights
        
        # 키 이름에서 'module.' 접두사 제거 (필요한 경우)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        # 모델의 state_dict와 로드된 state_dict의 키 비교
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
        
        if len(pretrained_dict) == len(model_dict):
            print("All parameters loaded successfully.")
        else:
            print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} parameters.")
        
        # 모델에 가중치 로드
        model.load_state_dict(pretrained_dict, strict=False)
        print(f"Successfully loaded VDSR weights from {weights_path}")
    else:
        print(f"=> no VDSR model found at '{weights_path}'")
    
    return model