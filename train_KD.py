import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from models import get_resnet18_cifar, get_resnet9_cifar
from dataset import get_dataloaders

# --- Hyperparameters ---
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.1
TEMPERATURE = 4.0
ALPHA = 0.7
TEACHER_WEIGHTS_PATH = "teacher.pth"
CHECKPOINT_PATH = "kd_student_checkpoint.pth"
BEST_MODEL_PATH = "kd_student.pth"
# -----------------------


def save_checkpoint_atomic(checkpoint, path):
    temp_path = path + ".tmp"
    torch.save(checkpoint, temp_path)
    os.replace(temp_path, path)


def kd_loss(student_logits, teacher_logits, targets, temperature, alpha):
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature ** 2)

    hard_loss = F.cross_entropy(student_logits, targets)
    total_loss = alpha * soft_loss + (1.0 - alpha) * hard_loss
    return total_loss, soft_loss, hard_loss


def train_kd():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Student with Knowledge Distillation on device: {device}")

    if not os.path.exists(TEACHER_WEIGHTS_PATH):
        raise FileNotFoundError(
            f"Teacher weights not found: {TEACHER_WEIGHTS_PATH}. Train the teacher first."
        )

    trainloader, testloader = get_dataloaders(BATCH_SIZE)

    teacher = get_resnet18_cifar().to(device)
    teacher.load_state_dict(torch.load(TEACHER_WEIGHTS_PATH, map_location=device))
    teacher.eval()

    student = get_resnet9_cifar().to(device)
    optimizer = optim.SGD(
        student.parameters(),
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
            student.load_state_dict(checkpoint["student_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            best_acc = checkpoint["best_acc"]
            start_epoch = checkpoint["epoch"] + 1
            print(f"Resumed from epoch {start_epoch} with best accuracy {best_acc:.2f}%")
        except Exception as error:
            print(f"Checkpoint load failed: {error}")
            print("Ignoring corrupted checkpoint and starting KD training from scratch.")
    else:
        print("No checkpoint found. Starting KD training from scratch.")

    for epoch in range(start_epoch, EPOCHS):
        student.train()
        total_loss = 0.0
        total_soft_loss = 0.0
        total_hard_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(trainloader, leave=False, desc=f"Epoch {epoch+1}/{EPOCHS} [KD Train]")
        for inputs, targets in loop:
            inputs, targets = inputs.to(device), targets.to(device)

            with torch.no_grad():
                teacher_outputs = teacher(inputs)

            optimizer.zero_grad()
            student_outputs = student(inputs)
            loss, soft_loss, hard_loss = kd_loss(
                student_outputs,
                teacher_outputs,
                targets,
                TEMPERATURE,
                ALPHA,
            )
            loss.backward()
            optimizer.step()

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_soft_loss += soft_loss.item() * batch_size
            total_hard_loss += hard_loss.item() * batch_size
            _, predicted = student_outputs.max(1)
            total += batch_size
            correct += predicted.eq(targets).sum().item()

            loop.set_postfix(
                loss=total_loss / total,
                soft=total_soft_loss / total,
                hard=total_hard_loss / total,
                acc=100.0 * correct / total,
            )

        scheduler.step()

        student.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = student(inputs)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()

        acc = 100.0 * test_correct / test_total
        print(f"Epoch {epoch+1}/{EPOCHS} | KD Student Test Acc: {acc:.2f}%")

        checkpoint = {
            "epoch": epoch,
            "student_state_dict": student.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_acc": best_acc,
            "temperature": TEMPERATURE,
            "alpha": ALPHA,
        }
        save_checkpoint_atomic(checkpoint, CHECKPOINT_PATH)

        if acc > best_acc:
            print(f"--> Accuracy improved from {best_acc:.2f}% to {acc:.2f}%. Saving kd_student.pth...")
            best_acc = acc
            torch.save(student.state_dict(), BEST_MODEL_PATH)
            checkpoint["best_acc"] = best_acc
            save_checkpoint_atomic(checkpoint, CHECKPOINT_PATH)

    print(f"KD Training Complete! Best KD Student Accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    train_kd()
