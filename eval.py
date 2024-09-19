import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from models.faster_rcnn_vdsr import FasterRCNN_VDSR
from data.dataset import SOD4SBDataset
from utils.transforms import sod4sb_transforms
from tqdm import tqdm
import os
import json
from inference_slicer import InferenceSlicer

def load_model(model_path, num_classes, vdsr_weights_path):
    model = FasterRCNN_VDSR(num_classes, vdsr_weights_path)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def perform_inference(model, image, device):
    model.to(device)
    image = image.to(device)
    
    # InferenceSlicer 설정
    slicer = InferenceSlicer(
        model,
        slice_size=(1024, 1024),
        overlap_size=(200, 200),
        batch_size=1,
        device=device
    )
    
    with torch.no_grad():
        predictions = slicer(image.unsqueeze(0))
    return predictions[0]

def visualize_result(image, prediction, threshold=0.5, output_path=None):
    image = image.cpu().permute(1, 2, 0).numpy()
    draw = ImageDraw.Draw(Image.fromarray((image * 255).astype('uint8')))

    boxes = prediction['boxes'][prediction['scores'] > threshold].cpu().numpy()
    labels = prediction['labels'][prediction['scores'] > threshold].cpu().numpy()
    scores = prediction['scores'][prediction['scores'] > threshold].cpu().numpy()

    for box, label, score in zip(boxes, labels, scores):
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], box[1]), f"Bird: {score:.2f}", fill="red")

    plt.figure(figsize=(20, 11))  # Adjust figure size for 3840x2160 images
    plt.imshow(image)
    plt.axis('off')
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()
    else:
        plt.show()

def calculate_map(predictions, targets):
    # Implement mAP calculation here
    # This is a placeholder and should be replaced with actual mAP calculation
    return 0.0

def main():
    model_path = 'path/to/your/best_model.pth'
    vdsr_weights_path = 'path/to/your/vdsr_weights.pth'
    test_data_root = 'path/to/test/images'
    test_ann_file = 'path/to/test/annotations.json'
    output_dir = 'inference_results'
    num_classes = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(output_dir, exist_ok=True)

    model = load_model(model_path, num_classes, vdsr_weights_path)
    model.to(device)

    test_dataset = SOD4SBDataset(
        root_dir=test_data_root,
        ann_file=test_ann_file,
        transform=sod4sb_transforms
    )

    all_predictions = []
    all_targets = []

    for idx in tqdm(range(len(test_dataset)), desc="Performing inference"):
        image, target = test_dataset[idx]
        file_name = test_dataset.images[idx]

        prediction = perform_inference(model, image, device)
        
        all_predictions.append(prediction)
        all_targets.append(target)

        output_path = os.path.join(output_dir, f"result_{file_name}")
        visualize_result(image, prediction, output_path=output_path)

    mAP = calculate_map(all_predictions, all_targets)
    print(f"Mean Average Precision (mAP): {mAP:.4f}")

    # Save results to a JSON file
    results = {
        "mAP": mAP,
        "predictions": [
            {
                "file_name": test_dataset.images[i],
                "boxes": pred['boxes'].cpu().numpy().tolist(),
                "scores": pred['scores'].cpu().numpy().tolist(),
                "labels": pred['labels'].cpu().numpy().tolist()
            }
            for i, pred in enumerate(all_predictions)
        ]
    }

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()