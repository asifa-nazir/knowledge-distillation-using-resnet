import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models import get_resnet9_cifar
from dataset import get_dataloaders

# --- Hyperparameters ---
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.1
CHECKPOINT_PATH = "student_checkpoint.pth"
BEST_MODEL_PATH = "student.pth"
# -----------------------


def save_checkpoint_atomic(checkpoint, path):
    temp_path = path + ".tmp"
    torch.save(checkpoint, temp_path)
    os.replace(temp_path, path)


def train_student():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Student (ResNet-9) on device: {device}")

    trainloader, testloader = get_dataloaders(BATCH_SIZE)

    model = get_resnet9_cifar().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=0.9,
        weight_decay=5e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0
    start_epoch = 0

    if os.path.exists(CHECKPOINT_PATH):
        try:
            print(f"Found checkpoint: {CHECKPOINT_PATH}. Resuming training...")
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            best_acc = checkpoint["best_acc"]
            start_epoch = checkpoint["epoch"] + 1
            print(f"Resumed from epoch {start_epoch} with best accuracy {best_acc:.2f}%")
        except Exception as error:
            print(f"Checkpoint load failed: {error}")
            print("Ignoring corrupted checkpoint and starting training from scratch.")
    else:
        print("No checkpoint found. Starting training from scratch.")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(trainloader, leave=False, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for inputs, targets in loop:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            loop.set_postfix(loss=train_loss / total, acc=100.0 * correct / total)

        scheduler.step()

        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()

        acc = 100.0 * test_correct / test_total
        print(f"Epoch {epoch+1}/{EPOCHS} | Test Acc: {acc:.2f}%")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_acc": best_acc,
        }
        save_checkpoint_atomic(checkpoint, CHECKPOINT_PATH)

        if acc > best_acc:
            print(f"--> Accuracy improved from {best_acc:.2f}% to {acc:.2f}%. Saving student.pth...")
            best_acc = acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            checkpoint["best_acc"] = best_acc
            save_checkpoint_atomic(checkpoint, CHECKPOINT_PATH)

    print(f"Training Complete! Best Student Accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    train_student()
