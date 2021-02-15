# MLFlow-augmented version of https://github.com/pytorch/examples/blob/master/mnist/main.py

from __future__ import print_function
import os
import mlflow
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from sklearn.metrics import classification_report

from src.git_autocommit import autocommit

TRACKING_URI = 'http://localhost:5000'
EXPERIMENT_NAME = 'mnist'


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    losses = []
    preds = []
    targets = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        preds.append(output.cpu().detach().numpy())
        targets.append(target.cpu().detach().numpy().reshape(-1, 1))

    losses = np.mean(losses)
    targets = np.vstack(targets)
    preds = np.vstack(preds)

    scores = classification_report(targets, preds.argmax(1), output_dict=True)
    scores['loss'] = losses

    df = pd.json_normalize(scores, sep='_')
    df = df.to_dict(orient='records')[0]
    return df


def test(model, device, test_loader):
    model.eval()
    losses = []
    preds = []
    targets = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target)
            losses.append(loss.item())
            preds.append(output.cpu().detach().numpy())
            targets.append(target.cpu().detach().numpy().reshape(-1, 1))

    losses = np.mean(losses)
    targets = np.vstack(targets)
    preds = np.vstack(preds)

    scores = classification_report(targets, preds.argmax(1), output_dict=True)
    scores['loss'] = losses

    df = pd.json_normalize(scores, sep='_')
    df = df.to_dict(orient='records')[0]
    return df


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    os.system("conda env export > environment.yml")
    autocommit(file_paths=['./'], message='Trying CNN')

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    with mlflow.start_run():
        mlflow.log_params(vars(args))
        for epoch in range(1, args.epochs + 1):
            train_metrics = train(args, model, device,
                               train_loader, optimizer, epoch)
            test_metrics = test(model, device, test_loader)
            print(test_metrics['accuracy'])
            scheduler.step()

            mlflow.log_metrics(train_metrics, epoch)
            mlflow.log_metrics(test_metrics, epoch)

        if args.save_model:
            torch.save(model.state_dict(), "mnist_cnn.pt")

if __name__ == '__main__':
    main()
