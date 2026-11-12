import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split, DataLoader
import time
import json

from common import confusion_matrix, get_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = F.log_softmax(model(data), dim=1)
            pred = output.data.max(1, keepdim=True)[1]
            correct += (pred.eq(target.data.view_as(pred)).sum().item())
    
    return 100.0 * correct / len(loader.dataset)

def save_confusion_matrix(matrix, filename):
    with open(filename, "w") as f:
        f.write("[\n")
        f.write(",\n".join(f"    {json.dumps(row)}" for row in matrix.tolist()))
        f.write("\n]\n")

def train(model, train_loader, test_loader, optimizer, loss_fn, epochs, 
            test_every=True, scheduler=None, log=False):
    if log:
        log = open("hw4_1_log.txt", "w")
    for epoch in range(1, epochs+1):
        loss_total = 0
        model.train()
        for _, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            logits = model(data)
            loss = loss_fn(logits, target)
            loss_total += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if scheduler:
            scheduler.step()
        
        if test_every or epoch == epochs:
            test_accuracy = round(test(model, test_loader), 3)
        else:
            test_accuracy = "..."

        avg_loss = loss_total / len(train_loader)
        if log:
            log.write(f"epoch {epoch} | average loss: {avg_loss:.3f}, test accuracy: {test_accuracy}\n")
        print(f"\repoch {epoch} | average loss: {avg_loss:.3f}, test accuracy: {test_accuracy}", end="")
    training_accuracy = round(test(model, train_loader), 3)
    print()
    if log:
        log.close()
    return training_accuracy, test_accuracy

def build_MLP(cfg):
    hidden = cfg.get("hidden", [256, 128])
    dropout = cfg.get("dropout", 0.0)
    activation_name = cfg.get("activation", "ReLU")
    activations = {"ReLU": nn.ReLU, "LeakyReLU": nn.LeakyReLU}
    activation = activations[activation_name]()
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
    torch.manual_seed(1)
    model = nn.Sequential(nn.Flatten(), nn.Linear(784, 64), nn.ReLU(), nn.Linear(64, 10)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    start_time = time.time()
    training_accuracy, test_accuracy = train(model, train_loader, test_loader, optimizer, 
                                            nn.CrossEntropyLoss(), 3, log=True)
    t = time.time() - start_time
    print(f"Finished training in {t:.2f} seconds.", end=" ")
    print(f"Training accuracy: {training_accuracy}, Test accuracy: {test_accuracy}\n")
    c_matrix = confusion_matrix(model, test_loader)
    print("Confusion Matrix:")
    print(c_matrix)
    save_confusion_matrix(c_matrix, "hw4_1a_confusion_matrix.json")
    print()

def part_b():
    print("Part b")
    train_loader, test_loader = get_data()
    full = train_loader.dataset
    train_subset, val_subset = random_split(full, [50000, 10000], 
                                            generator=torch.Generator().manual_seed(0))
    val_subset_loader = DataLoader(val_subset, batch_size=64)

    baseline = {"hidden": [128], "epochs": 10, "lr": 0.001, "dropout": 0.0}
    configs = [
        ("baseline", baseline),
        ("width_64", {**baseline, "hidden": [64]}),
        ("width_256", {**baseline, "hidden": [256]}),
        ("depth_2", {**baseline, "hidden": [128, 128]}),
        ("epochs_5", {**baseline, "epochs": 5}),
        ("epochs_15", {**baseline, "epochs": 15}),
        ("lr_0.01", {**baseline, "lr": 0.01}),
        ("lr_0.0001", {**baseline, "lr": 0.0001}),
        ("dropout_0.2", {**baseline, "dropout": 0.2}),
    ]
    results = []

    for experiment, cfg in configs:
        torch.manual_seed(1)
        model = build_MLP(cfg).to(device)
        lr = cfg.get("lr", 0.001)
        epochs = cfg.get("epochs", 10)
        print(f"trying {experiment}: {cfg}")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        run_train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, 
                                    generator=torch.Generator().manual_seed(1))
        training_accuracy, val_accuracy = train(model, run_train_loader, val_subset_loader, optimizer, 
                                                nn.CrossEntropyLoss(), epochs, False)
        print(f"Training accuracy: {training_accuracy}, Validation accuracy: {val_accuracy}\n")
        results.append({"experiment": experiment, "config": cfg, "validation_accuracy": val_accuracy})
    results.sort(key = lambda x: x["validation_accuracy"], reverse = True)
    with open("hw4_1b_results.json", "w") as f:
        json.dump(results, f, indent=4)

    best_model = build_MLP(results[0]["config"]).to(device)
    optimizer = torch.optim.Adam(best_model.parameters(), lr=results[0]["config"].get("lr", 0.001))
    epochs = results[0]["config"].get("epochs", 10)
    start_time = time.time()
    training_accuracy, test_accuracy = train(best_model, train_loader, test_loader, optimizer, nn.CrossEntropyLoss(), epochs, False)
    print(f"Finished training best model in {time.time()-start_time:.2f} seconds.", end=" ")
    print(f"Training accuracy: {training_accuracy}, Test accuracy: {test_accuracy}\n")
    c_matrix = confusion_matrix(best_model, test_loader)
    save_confusion_matrix(c_matrix, "hw4_1b_confusion_matrix.json")
    print()


def part_c():
    print("Part c")
    train_loader, test_loader = get_data()
    torch.manual_seed(1)
    model = nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.1),
        nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.1),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.1),
        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.Flatten(), nn.Linear(128*3*3, 10)
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.3)
    start_time = time.time()
    training_accuracy, test_accuracy = train(model, train_loader, test_loader, optimizer, 
        nn.CrossEntropyLoss(), 15, test_every=False, scheduler=scheduler)
    print(f"Finished training in {time.time()-start_time:.2f} seconds. Training accuracy: {training_accuracy}, Test accuracy: {test_accuracy}\n")
    c_matrix = confusion_matrix(model, test_loader)
    save_confusion_matrix(c_matrix, "hw4_1c_confusion_matrix.json")

if __name__ == "__main__":
    part_a()
    part_b()
    part_c()