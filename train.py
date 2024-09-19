import torch
from torch.utils.data import DataLoader
from models.faster_rcnn_vdsr import FasterRCNN_VDSR
from data.dataset import SOD4SBDataset
from utils.transforms import sod4sb_transforms
from tqdm import tqdm
import time
import os

def train_model(model, data_loader, optimizer, num_epochs, device, save_dir, save_interval=1):
    model.to(device)
    best_loss = float('inf')
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for i, (images, targets) in enumerate(progress_bar):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
            
            progress_bar.set_postfix({
                'loss': f"{losses.item():.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
            })
        
        avg_loss = epoch_loss / len(data_loader)
        end_time = time.time()
        epoch_time = end_time - start_time
        
        print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")
        
        if (epoch + 1) % save_interval == 0:
            save_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to {best_model_path}")

if __name__ == "__main__":
    # 하이퍼파라미터 및 설정
    num_classes = 2  # 배경 + 새
    vdsr_weights_path = '/home/hong/Faster R-CNN with VDSR/models/model_epoch_50.pth'  # VDSR 가중치 파일 경로
    batch_size = 2
    num_epochs = 10
    learning_rate = 0.0001
    momentum = 0.9
    weight_decay = 0.0005
    save_dir = 'model_weights'
    save_interval = 1

    # 데이터셋 및 데이터 로더 설정
    dataset = SOD4SBDataset(
        root_dir='/home/hong/Faster R-CNN with VDSR/data/dataset/images',
        ann_file='/home/hong/Faster R-CNN with VDSR/data/dataset/annotations/split_train_coco.json',
        transform=sod4sb_transforms,
        target_size=(960, 540)
    )
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    # 모델 초기화
    model = FasterRCNN_VDSR(num_classes, vdsr_weights_path)

    # 옵티마이저 설정
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 학습 실행
    train_model(model, data_loader, optimizer, num_epochs, device, save_dir, save_interval)

print("Training completed.")