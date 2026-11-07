import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def make_loaders():

    # Transform the images into tensors
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # load the training images and assign classes
    base_ds = datasets.ImageFolder(
        "data/train",
        transform = transform
    )

    # load in the test images and assign classes
    test_ds = datasets.ImageFolder(
        "data/test",
        transform = transform
    )

    # Split the training set (80,000 training, 20,000 validation
    seed = 42
    g = torch.Generator().manual_seed(seed)
    train_size = int(0.8 * len(base_ds))
    val_size = len(base_ds) - train_size

    train_subset, val_subset = random_split(
        base_ds,
        [train_size, val_size],
        generator = g
    )

    # Data Loaders
    batch_size = 32
    num_workers = 2
    pin = torch.cuda.is_available()
    g_1 = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_subset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers,
        generator = g_1,
        pin_memory = pin
    )
    val_loader = DataLoader(
        val_subset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
        pin_memory = pin
    )
    test_loader = DataLoader(
        test_ds,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
        pin_memory = pin
    )

    return train_loader, val_loader, test_loader


def main():
    train_loader, val_loader, test_loader = make_loaders()

    print("Training Batches:", len(train_loader))
    print("Validation Batches:", len(val_loader))
    print("Test Batches:", len(test_loader))

if __name__ == "__main__":
    main()
