import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import time
import numpy as np
import json

def accuracy(model, X, y):
    preds = np.argmax(model(X), axis=1)
    return np.mean(preds == y)

def confusion_matrix(y_true, y_pred):
    C = np.zeros((10, 10))
    for (t, p) in zip(y_true, y_pred):
        C[t][p] += 1
    return C

def get_data():
    data_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.KMNIST(root='./datasets', train=True, download=True, transform=data_transforms)
    test_dataset = torchvision.datasets.KMNIST(root='./datasets', train=False, download=True, transform=data_transforms)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_loader, test_loader

def test(model, loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            output = F.log_softmax(model(data), dim=1)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += (pred.eq(target.data.view_as(pred)).sum().item())
            total = total + 1
    
    return 100.0 * correct / len(loader.dataset)

def train(model, train_loader, test_loader, optimizer, loss_fn, epochs):
    for epoch in range(1, epochs+1):
        for _, (data, target) in enumerate(train_loader):
            logits = model(data)
            loss = loss_fn(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        acc = test(model, test_loader)
        print(f"epoch {epoch} | loss: {loss:.3f}, test accuracy: {acc:.3f}")
    return acc

def build_MLP(cfg):
    hidden = cfg.get("hidden", [256, 128])
    dropout = cfg.get("dropout", 0.0)
    activation = cfg.get("activation", nn.ReLU())
    layers = [nn.Flatten()]
    input_dim = 784
    for width in hidden:
        layers.extend([nn.Linear(input_dim, width), activation])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        input_dim = width
    layers.append(nn.Linear(input_dim, 10))
    return nn.Sequential(*layers)

def part_a():
    print("Part a")
    train_loader, test_loader = get_data()
    
    # part a
    model = nn.Sequential(nn.Flatten(), nn.Linear(784, 64), nn.ReLU(), nn.Linear(64, 10))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    start_time = time.time()
    train(model, train_loader, test_loader, optimizer, nn.CrossEntropyLoss(), 3)
    
    print(f"Finished training in {time.time()-start_time:.2f} seconds\n")


def part_b():
    print("Part b")
    train_loader, test_loader = get_data()
    configs = [
        {"hidden": [128],        "epochs": 10, "lr": 0.001},
        {"hidden": [256, 128],   "epochs": 15, "lr": 0.001},
        {"hidden": [512, 256],   "epochs": 20, "lr": 0.001, "dropout": 0.2},
        {"hidden": [256, 128],   "activation": nn.LeakyReLU(), "epochs": 20, "lr": 0.001}
    ]
    results = []

    for cfg in configs:
        model = build_MLP(cfg)
        lr = cfg.get("lr", 0.001)
        epochs = cfg.get("epochs", 15)
        print(f"trying config with {epochs} epochs, ")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        acc = train(model, train_loader, test_loader, optimizer, nn.CrossEntropyLoss(), epochs)
        results.append({"config":cfg, "accuracy":acc})
    results.sort(key = lambda x: x["accuracy"], reverse = True)
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
    print()

    


def main():
    part_a()
    part_b()

if __name__ == "__main__":
    main()