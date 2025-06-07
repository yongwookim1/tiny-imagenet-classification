import os
import time
from pathlib import Path
import random
import numpy as np
from livelossplot import PlotLosses


import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms


def set_seed():
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_device():
    if torch.cuda.is_available():
        device = "cuda:1"
    elif torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = "cpu"
    return device


def set_dataloaders(dataset, batch_size):
    if dataset == "tiny-224":
        data_dir = "tiny-224"
    else:
        data_dir = "tiny-imagenet-200"

    num_workers = {"train": 8, "val": 8, "test": 0}
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ]
        ),
    }
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ["train", "val", "test"]
    }
    dataloaders = {
        x: data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers[x])
        for x in ["train", "val", "test"]
    }
    return dataloaders


def set_model(model: str):
    # Load model
    torch.manual_seed(42)
    model_dict = {"resnet": models.resnet18(weights="DEFAULT"),
                  "efficientnet": models.efficientnet_b1(weights="DEFAULT"),
                  "widerenset": models.wide_resnet50_2(weights="DEFAULT"),
                  "mobilenet": models.mobilenet_v3_small(weights="DEFAULT"),
                  "vit": models.vit_b_16(weights="DEFAULT"),
                  "swin": models.swin_t(weights="DEFAULT"),
                  }
    model_ft = model_dict[model]

    # Finetune Final few layers to adjust for tiny imagenet input
    try:
        num_features = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_features, 200)
    except:
        try:
            num_features = model_ft.classifier[-1].in_features
            model_ft.classifier[-1] = nn.Linear(num_features, 200)
        except:
            try:
                num_features = model_ft.head.in_features
                model_ft.head = nn.Linear(num_features, 200)
            except:
                num_features = model_ft.heads[-1].in_features
                model_ft.heads[-1] = nn.Linear(num_features, 200)                

    # Loss Function
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer_ft = optim.AdamW(model_ft.parameters(), lr=1e-3, weight_decay=1e-2)
    return model_ft, criterion, optimizer_ft


def load_model(model, path):
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)
    return model


def train_model(output_path, model, dataloaders, criterion, optimizer, device, num_epochs=5, scheduler=None) -> int:
    (Path("models") / output_path).mkdir(parents=True, exist_ok=True)
    since = time.time()
    liveloss = PlotLosses()

    best_acc = 0.0
    best = 0

    for epoch in range(num_epochs):
        logs = {}
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                if scheduler != None:
                    scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # statistics
                _, preds = torch.max(outputs, 1)
                running_loss += loss.detach() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.float() / len(dataloaders[phase].dataset)

            if phase == "train":
                prefix = ""
            else:
                prefix = "val_"

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best = epoch + 1

            logs[prefix + "log loss"] = epoch_loss.item()
            logs[prefix + "accuracy"] = epoch_acc.item()

        liveloss.update(logs)
        liveloss.send()

        torch.save(model.state_dict(), f"./models/{output_path}/{epoch + 1}_epoch.pt")
    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best Validation Accuracy: {}, Epoch: {}".format(best_acc, best))
    return best


def test_model(model, dataloaders, criterion, device):
    since = time.time()
    phase = "test"

    # Each epoch has a training and validation phase
    model.eval()  # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders[phase]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloaders[phase].dataset)
    epoch_acc = running_corrects.float() / len(dataloaders[phase].dataset)

    time_elapsed = time.time() - since
    print("Test Loss: {:.4f} Acc: {:.4f}".format(epoch_loss, epoch_acc))
    print("Test complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))