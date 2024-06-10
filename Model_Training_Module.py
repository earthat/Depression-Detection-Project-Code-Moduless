# %%
import pickle
import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class EEGSignalDataset(Dataset):
    def __init__(self, training_data):
        training_data_transformed = self._transform(training_data)
        self.signals, self.Adj_dist_matrices, self.y = training_data_transformed

    def _transform(self, training_data):
        signals, Adj_dist_matrices, labels = [], [], []
        for signal, Adj_dist_matrix, label in training_data:
            torch_signal = torch.from_numpy(signal).float()
            torch_Adj_dist_matrix = torch.from_numpy(Adj_dist_matrix).float()
            label = float(label)
            signals.append(torch_signal) 
            Adj_dist_matrices.append(torch_Adj_dist_matrix) 
            labels.append(label)
        return signals, Adj_dist_matrices, labels

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.signals[idx], self.Adj_dist_matrices[idx], self.y[idx]


def train_test_datasets(dataset, batch_size, train_test_split):
    train_size = int(train_test_split * len(dataset))
    test_size = len(dataset) - train_size
    trainDataset, testDataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_dataset = EEGSignalDataset(trainDataset)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = EEGSignalDataset(testDataset)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader


def train_loop(dataloader, model, loss_fn, optimizer, batch_size):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X_Signal, X_AdjMatrixy, y) in enumerate(dataloader):

        # Compute prediction and loss
        pred = model(X_Signal, X_AdjMatrixy)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X_Signal)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():

        def fn_bool(p, y, treshold):
            result = y*p+(1-y)*(1-p)
            return result > treshold

        for X_Signal, X_AdjMatrixy, y in dataloader:
            pred = model(X_Signal, X_AdjMatrixy)
            test_loss += loss_fn(pred, y).item()
            probability = nn.functional.sigmoid(pred)
            correct += (fn_bool(probability, y, 0.5)).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

