import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json
from torchvision.transforms import functional as F

class SOD4SBDataset(Dataset):
    def __init__(self, root_dir, ann_file, transform=None, target_size=(960, 540)):
        self.root_dir = root_dir
        self.transform = transform
        self.target_size = target_size
        
        # JSON 어노테이션 파일 로드
        with open(ann_file, 'r') as f:
            anno = json.load(f)
        
        self.annotations = anno['annotations']
        self.images = anno['images']
        
        # 이미지 ID를 키로 하는 딕셔너리 생성
        self.image_id_to_annotations = {}
        for ann in self.annotations:
            image_id = ann['image_id']
            if image_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[image_id] = []
            self.image_id_to_annotations[image_id].append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        
        # 이미지 리사이징
        image = F.resize(image, self.target_size)
        
        # 어노테이션 정보 가져오기
        image_id = img_info['id']
        anns = self.image_id_to_annotations.get(image_id, [])
        
        boxes = []
        for ann in anns:
            x_min, y_min, width, height = ann['bbox']
            x_max, y_max = x_min + width, y_min + height
            boxes.append([x_min, y_min, x_max, y_max])
        
        # 박스가 없는 경우 처리
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            
            # 박스 좌표 리사이징
            orig_w, orig_h = img_info['width'], img_info['height']
            scale_x, scale_y = self.target_size[0] / orig_w, self.target_size[1] / orig_h
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            
            labels = torch.ones((boxes.shape[0],), dtype=torch.int64)  # 모든 객체는 새(클래스 1)입니다.
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        
        if self.transform:
            image, target = self.transform(image, target)
        
        return image, target
