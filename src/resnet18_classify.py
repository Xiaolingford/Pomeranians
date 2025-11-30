import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os

# ---------------------------------------------------------
#  Config
# ---------------------------------------------------------
DATA_DIR = r"C:\Users\User\Desktop\Programming languages for vs\T\Thesis_Microplastics\dataC"
NUM_CLASSES = 2
BATCH_SIZE = 8
LR = 2e-4
EPOCHS = 20
SAVE_PATH = "resnet18_best.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------
#  Data augmentation and loading
# ---------------------------------------------------------
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# ---------------------------------------------------------
#  Training function
# ---------------------------------------------------------
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# ---------------------------------------------------------
#  Validation function
# ---------------------------------------------------------
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    loss = running_loss / total
    acc = correct / total
    return loss, acc


# ---------------------------------------------------------
#  Main training script
# ---------------------------------------------------------
if __name__ == '__main__':
    print("Using device:", device)
    
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tf)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_tf)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print("Classes:", train_dataset.classes)

    # Model setup
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Training Loop
    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"\nEpoch {epoch}/{EPOCHS}")
        print(f" Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f" Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc*100:.2f}%")

        # save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"✅ Saved best model so far to {SAVE_PATH}")

    print("\nTraining complete!")
    print(f"Best Validation Accuracy: {best_acc*100:.2f}%")