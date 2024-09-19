import torch
import torch.nn as nn
import torchvision.ops as ops
from .faster_rcnn import get_faster_rcnn_model
from .vdsr import Net, load_vdsr_weights

class FasterRCNN_VDSR(nn.Module):
    def __init__(self, num_classes, vdsr_weights_path=None):
        super(FasterRCNN_VDSR, self).__init__()
        self.faster_rcnn = get_faster_rcnn_model(num_classes)
        self.vdsr = Net()
        
        if vdsr_weights_path:
            self.vdsr = load_vdsr_weights(self.vdsr, vdsr_weights_path)

    def forward(self, images, targets=None):
        if self.training:
            return self.faster_rcnn(images, targets)

        detections = self.faster_rcnn(images)
        
        for i, detection in enumerate(detections):
            boxes = detection['boxes']
            scores = detection['scores']
            labels = detection['labels']
            
            for j, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                x1, y1, x2, y2 = box.int()
                w, h = x2 - x1, y2 - y1
                
                # 박스의 종횡비 계산
                aspect_ratio = w / h if h != 0 else 1
                
                # 종횡비가 극단적인 경우 (예: 2:1 또는 1:2 이상)
                if aspect_ratio > 2 or aspect_ratio < 0.5:
                    # 박스 주변을 포함하여 정사각형 영역 추출
                    size = max(w, h)
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    new_x1 = max(0, center_x - size // 2)
                    new_y1 = max(0, center_y - size // 2)
                    new_x2 = min(images[i].shape[2], new_x1 + size)
                    new_y2 = min(images[i].shape[1], new_y1 + size)
                    roi = images[i][:, new_y1:new_y2, new_x1:new_x2]
                else:
                    roi = images[i][:, y1:y2, x1:x2]
                
                # VDSR 적용
                enhanced_roi = self.vdsr(roi.unsqueeze(0)).squeeze(0)
                
                # 향상된 ROI를 원본 이미지에 다시 삽입
                if aspect_ratio > 2 or aspect_ratio < 0.5:
                    # 원래 박스 크기에 맞게 리사이즈
                    enhanced_roi = ops.interpolate(enhanced_roi.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).squeeze(0)
                    images[i][:, y1:y2, x1:x2] = enhanced_roi
                else:
                    images[i][:, y1:y2, x1:x2] = enhanced_roi

        return detections