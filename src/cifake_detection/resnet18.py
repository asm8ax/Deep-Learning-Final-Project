import argparse
import time
import torch
from torch import nn
import torch.nn.functional as F
from .data_loader import make_loaders
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, use_skip=True):
        super().__init__()

        # first Conv
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias= False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu= nn.ReLU(inplace=True)

        #second Conv
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias= False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # skip connection
        self.use_skip = use_skip
        if use_skip:
            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_skip:
            out = out + self.shortcut(x)
        return self.relu(out)


class ResNet18(nn.Module):
    def __init__(self, num_classes = 1, use_skip=True):
        super().__init__()
        self.in_channels = 64
        self.use_skip = use_skip

        # stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1 , padding=1, bias= False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu= nn.ReLU(inplace=True)

        # layers
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride, self.use_skip))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out= self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)

def load_imagenet_layers(model):
    import torchvision
    tv = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    # keep only layer1-layer4; rename torchvision's "downsample" to our "shortcut"
    sd = {k.replace("downsample", "shortcut"): v
          for k, v in tv.state_dict().items() if k.startswith("layer")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("Left random (not loaded):", missing)
    print("Unexpected keys:", unexpected)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        # Load the batch and transfer to the GPU
        # inputs: [32, 3, 32, 32] (batch, features, pixels, pixels
        # labels [32] (batch size)
        inputs, labels = inputs.to(device), labels.to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Forward Pass
        outputs = model(inputs)
        labels = labels.float().unsqueeze(1)

        # calculate loss
        loss = criterion(outputs, labels)

        # Backpropogation
        loss.backward()

        # Update Weights
        optimizer.step()

        # Track Metrics
        running_loss += loss.item() * inputs.size(0)
        predicted = (outputs >=0).float()
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion, device):
    # Put model into evaluation mode
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad(): # turn off gradients
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward Pass
            outputs = model(inputs)
            labels = labels.float().unsqueeze(1)

            # calculate loss
            loss = criterion(outputs, labels)

            # Track Metrics
            running_loss += loss.item() * inputs.size(0)
            predicted = (outputs >=0).float()
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy().flatten())
            all_predictions.extend(predicted.cpu().numpy().flatten())

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)

    return epoch_loss, epoch_acc, precision, recall, f1

def plot_training_history(history, title = "Training History"):
    """Plot training and validation loss/accuracy."""

    epochs = range(1, len(history['train_loss']) + 1)

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history['train_loss'], label='Train Loss')
    ax1.plot(epochs, history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{title} - Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history['train_acc'], label='Train Acc')
    ax2.plot(epochs, history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'{title} - Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f"training_history_for_{title}.png")
    plt.close()

def measure_inference_ms(model, device, batch_size, n_runs=200, warmup=20):
    model.eval()
    x = torch.randn(batch_size, 3, 32, 32, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(n_runs):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.time() - start) / n_runs * 1000

def main(seed=42, variant="resnet", lr=0.001):
    torch.manual_seed(seed)
    if variant == "plain":
        NAME = "plain18"
    elif variant == "pretrained":
        NAME = f"pretrained18_lr{lr:g}"
    else:
        NAME = "resnet18"
    ckpt = f"{NAME}_seed{seed}_best.pth"

    # load the data from the other script
    train_loader, val_loader, test_loader = make_loaders()

    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Seed: {seed}")
    print(f"Variant: {variant}")
    print(f"Learning rate: {lr}")

    # model
    model = ResNet18(use_skip=(variant != "plain")).to(device)
    if variant == "pretrained":
        load_imagenet_layers(model)

    # Binary Class Loss
    criterion = nn.BCEWithLogitsLoss()

    # Adam Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    # epochs
    num_epochs = 10

    # keep track of metrics throughout run
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': []
    }
    best_val_acc = 0.0

    # loop over epochs and train the model
    total_start = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 30)

        # training metrics
        epoch_start = time.time()
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )
        train_time = time.time() - epoch_start

        # validation metrics
        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        # append metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)

        # save the best model (based on validation accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt)
            print(f"New best model saved. Validation Accuracy: {best_val_acc:.2f}%")

        # Print loss/accuracy for each epoch
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Val Precision: {val_precision:.4f}')
        print(f'Val Recall: {val_recall:.4f}')
        print(f'Val F1: {val_f1:.4f}')
        print(f'Train time: {train_time:.1f}s')
        if device.type == "cuda":
            print(f'Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB')

    print(f"\nTotal training time: {(time.time() - total_start) / 60:.1f} min")
    # best model is saved, load it to gather final metrics
    model.load_state_dict(
        torch.load(
            ckpt,
            map_location=device,
            weights_only=True
        )
    )
    print(f"\nLoaded model with validation accuracy: {best_val_acc:.2f}%")

    # evaluate the best model on the training, evaluation, and test sets
    train_metrics = evaluate(model, train_loader, criterion, device)
    val_metrics = evaluate(model, val_loader, criterion, device)
    test_metrics = evaluate(model, test_loader, criterion, device)

    # Print final metrics
    print("\nFinal Model Performance")
    print("-" * 85)
    print(f"{'Dataset':<12} {'Loss':>10} {'Accuracy':>12} {'Precision':>12} {'Recall':>12} {'F1':>12}")
    print("-" * 85)

    for name, metrics in [
        ("Train", train_metrics),
        ("Validation", val_metrics),
        ("Test", test_metrics)
    ]:
        loss, acc, precision, recall, f1 = metrics

        print(f"{name:<12} {loss:>10.4f} {acc:>11.2f}% {precision:>12.4f} {recall:>12.4f} {f1:>12.4f}")

    t1 = measure_inference_ms(model, device, batch_size=1)
    t32 = measure_inference_ms(model, device, batch_size=32)
    print(f"\nInference time: {t1:.2f} ms/image (batch 1), {t32 / 32:.3f} ms/image (batch 32)")

    # plot training curves
    plot_training_history(history, title=f"{NAME.capitalize()}_seed{seed}")

    return history

def cli():
    parser = argparse.ArgumentParser(description="Train a CIFAKE ResNet variant")
    parser.add_argument("seed", nargs="?", type=int, default=42)
    parser.add_argument("variant", nargs="?", choices=["resnet", "plain", "pretrained"], default="resnet")
    parser.add_argument("lr", nargs="?", type=float, default=0.001)
    args = parser.parse_args()
    main(args.seed, args.variant, args.lr)


if __name__ == "__main__":
    cli()
