import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@torch.no_grad()
def predict(model, loader):
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).argmax(1).cpu()); labels.append(y)
    return torch.cat(preds).numpy(), torch.cat(labels).numpy()

def confusion_matrix(model, loader):
    preds, labels = predict(model, loader)
    C = np.zeros((10, 10), dtype=int)
    np.add.at(C, (labels, preds), 1)
    return C

def get_data():
    data_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.KMNIST(root='./datasets', train=True, download=True, transform=data_transforms)
    test_dataset = torchvision.datasets.KMNIST(root='./datasets', train=False, download=True, transform=data_transforms)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_loader, test_loader

def get_data_sets():
    train_loader, test_loader = get_data()
    train_loader = DataLoader(train_loader.dataset, batch_size=64, shuffle=False)

    X_train, Y_train, X_test, Y_test = [], [], [], []
    for data, target in train_loader:
        X_train.append(data.view(data.size(0), -1).numpy())
        Y_train.append(target.numpy())
    X_train = np.concatenate(X_train, axis=0)
    Y_train = np.concatenate(Y_train, axis=0)

    for data, target in test_loader:
        X_test.append(data.view(data.size(0), -1).numpy())
        Y_test.append(target.numpy())
    X_test = np.concatenate(X_test, axis=0)
    Y_test = np.concatenate(Y_test, axis=0)

    return X_train, Y_train, X_test, Y_test