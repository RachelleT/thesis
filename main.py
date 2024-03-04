from Preprocessing.preprocessing import Preprocessing
from Preprocessing.dataset import Dataset
from Models.classifier import ResNet
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.model_selection import KFold


def training_classifier(train_dataset):
    # Define the number of folds and batch size
    k_folds = 3
    batch_size = 64

    # Define the device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the k-fold cross validation
    kf = KFold(n_splits=k_folds, shuffle=True)

    # Loop through each fold
    for fold, (train_idx, test_idx) in enumerate(kf.split(train_dataset)):
        print(f"Fold {fold + 1}")
        print("-------")

        # Define the data loaders for the current fold
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(train_idx),
        )
        test_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(test_idx),
        )

        # Initialize the model and optimizer
        model = ResNet.ResNet50(10, 1).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Train the model on the current fold
        for epoch in range(1, 11):
            # train(model, device, train_loader, optimizer, epoch)
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = nn.functional.nll_loss(output, target)
                loss.backward()
                optimizer.step()

        # Evaluate the model on the test set
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += nn.functional.nll_loss(output, target, reduction="sum").item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100.0 * correct / len(test_loader.dataset)

        # Print the results for the current fold
        print(
            f"Test set: Avg loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n")


if __name__ == '__main__':
    # ResNet

    # Preprocessing
    raw_data = Preprocessing()
    train, test = raw_data.split_data()

    # Apply Transforms
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
    ])

    training_data = Dataset(train, transform_train)

    # Train Classifier
    training_classifier(training_data)
