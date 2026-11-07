import torch
from torch import nn
from .data_loader import make_loaders
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Input image is [32, 3, 32, 32]

        layers = []

        # first conv layer, [32, 32, 32, 32]
        layers.append(
            nn.Conv2d(
                in_channels = 3,
                out_channels = 32,
                kernel_size = 3,
                padding = 1,
                bias = True
            )
        )
        # non-linear activation
        layers.append(nn.ReLU(inplace=True))
        # max-pooling, [32, 32, 16, 16], reduces spacial dimension
        layers.append(nn.MaxPool2d(kernel_size=2))

        # second conv layer
        # input [32, 32, 16, 16]
        layers.append(
            nn.Conv2d(
                in_channels = 32,
                out_channels = 32,
                kernel_size = 3,
                padding = 1,
                bias = True
            )
        )
        # non-linear activation
        layers.append(nn.ReLU(inplace=True))
        # max pooling, [32, 32, 8, 8]
        layers.append(nn.MaxPool2d(kernel_size=2))

        # flatten, [32, 32, 8, 8] -> [32, 2,048]
        layers.append(nn.Flatten(1))

        # Output: Linear layer that produces prediction
        layers.append(nn.Linear(2048, 1))

        # Sequential that connects the layers
        self.sequential = nn.Sequential(*layers)

    def forward(self, x):
        # run the images through the network
        return self.sequential(x)

# define the function to send us through the CNN neural network
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


def main():
    torch.manual_seed(42)

    # load the data from the other script
    train_loader, val_loader, test_loader = make_loaders()

    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # model
    model = CNN().to(device)

    # Binary Class Loss
    criterion = nn.BCEWithLogitsLoss()

    # Adam Optemizer
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

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
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 30)

        # training metrics
        train_loss, train_acc = train_epoch(
            model, 
            train_loader, 
            criterion, 
            optimizer, 
            device
        )

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
            torch.save(model.state_dict(), "cnn_baseline_best.pth")
            print(f"New best model saved. Validation Accuracy: {best_val_acc:.2f}%")

        # Print loss/accuracy for each epoch
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Val Precision: {val_precision:.4f}')
        print(f'Val Recall: {val_recall:.4f}')
        print(f'Val F1: {val_f1:.4f}')

    # best model is saved, load it to gather final metrics
    model.load_state_dict(
        torch.load(
            "cnn_baseline_best.pth",
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

    # plot training curves
    plot_training_history(history, title="CNN_Baseline")

    return history

def cli():
    main()


if __name__ == "__main__":
    cli()
