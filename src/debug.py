from dataset import get_yolo_dataloaders
import torch

if __name__ == '__main__':
    train_loader, val_loader = get_yolo_dataloaders(
        train_root="./data_yolo_preprocessed",
        val_root="./data_yolo",
        batch_size=4,
        img_size=224,
        num_workers=0  # ✅ Set to 0 to avoid multiprocessing issues
    )

    # Check 10 batches
    print("=== Checking Training Data ===")
    for i, (imgs, targets) in enumerate(train_loader):
        if i >= 10:
            break
        
        for j, tgt in enumerate(targets):
            boxes = tgt['boxes']
            labels = tgt['labels']
            
            print(f"\nBatch {i}, Image {j}:")
            print(f"  Num boxes: {len(boxes)}")
            
            if len(boxes) > 0:
                print(f"  Box range: [{boxes.min():.3f}, {boxes.max():.3f}]")
                print(f"  Labels: {labels.tolist()}")
                
                # Check for invalid boxes
                x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                invalid = (x2 <= x1) | (y2 <= y1) | (boxes < 0).any(dim=1) | (boxes > 1).any(dim=1)
                if invalid.any():
                    print(f"  ⚠️ INVALID BOXES: {boxes[invalid]}")
            else:
                print(f"  No boxes in this image")
    
    print("\n=== Done checking! ===")