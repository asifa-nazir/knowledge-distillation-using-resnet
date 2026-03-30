import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models import get_resnet18_cifar
from dataset import get_dataloaders

# --- Hyperparameters ---
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.1
BEST_MODEL_PATH = "teacher.pth"
# -----------------------

def train_teacher():
    # 1. Setup Device (Will automatically use your GPU if you have one, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Teacher (ResNet-18) on device: {device}")

    # 2. Load Data
    trainloader, testloader = get_dataloaders(BATCH_SIZE)

    # 3. Initialize Model, Loss function, and Optimizer
    model = get_resnet18_cifar().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Standard optimizer settings for CIFAR-10
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    
    # Scheduler gradually reduces learning rate over epochs following a cosine curve
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0

    # 4. Main Training Loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        # tqdm creates a professional-looking progress bar
        loop = tqdm(trainloader, leave=False, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for inputs, targets in loop:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()           # Clear old gradients
            outputs = model(inputs)         # Forward pass
            loss = criterion(outputs, targets) # Calculate loss
            loss.backward()                 # Backward pass (calculate gradients)
            optimizer.step()                # Update weights

            # Calculate metrics
            train_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            loop.set_postfix(loss=train_loss/total, acc=100.*correct/total)
        
        # Step the learning rate scheduler after every epoch
        scheduler.step()

        # 5. Evaluation Loop (Test on unseen data)
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad(): # No gradients needed for evaluation (saves memory/speed)
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()

        acc = 100. * test_correct / test_total
        print(f"Epoch {epoch+1}/{EPOCHS} | Test Acc: {acc:.2f}%")

        # 6. Save the model ONLY if it improves
        if acc > best_acc:
            print(f"--> Accuracy improved from {best_acc:.2f}% to {acc:.2f}%. Saving teacher.pth...")
            best_acc = acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)

    print(f"Training Complete! Best Teacher Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    train_teacher()
