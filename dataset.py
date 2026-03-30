import torch
import torchvision
import torchvision.transforms as transforms

def get_dataloaders(batch_size=128):
    """
    Downloads CIFAR-10 and applies standard augmentations.
    """
    # 1. Training Augmentations
    # Pad the 32x32 image by 4 pixels, take a random 32x32 crop, flip randomly.
    # The normalization values are the exact statistical mean/std of CIFAR-10.
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 2. Testing Transforms
    # NO augmentation during testing. Only convert to tensor and normalize.
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 3. Download and load datasets
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader