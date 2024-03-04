import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Callable
from tqdm.notebook import tqdm, trange
from pathlib import Path

import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.utils.data.dataset import Subset
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import f1_score

EPOCHS = 2
KFOLDS = 2
SEED = 1999
BATCH_SIZE = 32
# model = torch.load(Path('E:\Prut\cxr\models\lung_segment_model_150823.pt'))
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('Using:', device)

def train(dataloader, model, loss_fn, optimizer, accuracy_fn):
    
    n_samples = 0
    train_loss = 0
    train_acc = 0

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device).float(), y.to(device).long()
        model = model.to(device)
        pred = model(X).float()
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_acc = accuracy_fn(y.int().cpu().numpy(), pred.argmax(1).cpu().numpy())

        n_samples += X.size(0)
        train_loss += loss.item() * X.size(0)
        train_acc += batch_acc * X.size(0)
        if batch % 100 == 0:
            print('Batch', batch, 'Out of', len(dataloader))
            print('Training loss', loss.item())
            print('Training acc', batch_acc)

    train_loss /= n_samples
    train_acc /= n_samples

    print('Train acc:', train_acc.item())

    return train_loss, train_acc

def test(dataloader, model, loss_fn, accuracy_fn):
    
    n_samples = 0
    test_loss = 0
    test_acc = 0

    model = model.to(device)
    model.eval()
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device).float(), y.to(device).long()

            pred = model(X).float()
            batch_loss = loss_fn(pred, y)
            batch_acc = accuracy_fn(y.int().cpu().numpy(), pred.argmax(1).cpu().numpy())

            n_samples += X.size(0)
            test_loss += batch_loss.item() * X.size(0)
            test_acc += batch_acc * X.size(0)


        test_loss /= n_samples
        test_acc /= n_samples

        print('Test acc:', test_acc.item())
    
    return test_loss, test_acc

def cross_validate(in_model, in_weights, dataset, k_folds=KFOLDS): # train:Callable=train, valid:Callable=test
    
    train_accuracies = []
    val_accuracies = []
    OUT = 2
    
    for i in range(k_folds):
        print('Fold:', i)

        model = in_model(weights=in_weights.IMAGENET1K_V1)
        model = model.to(device)
        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False
        
        # Change last layer to binary
        if model.__class__.__name__ in ['ConvNeXt', 'EfficientNet', 'VGG']:
            model.classifier[-1] = nn.Linear(in_features=model.classifier[-1].in_features, out_features=OUT, bias=True)
        elif model.__class__.__name__ in ['ResNet', 'Inception3']:
            model.fc = nn.Linear(in_features=model.fc.in_features, out_features=OUT, bias=True)
        elif model.__class__.__name__ in ['VisionTransformer']:
            model.heads[-1] = nn.Linear(in_features=model.heads[-1].in_features, out_features=OUT, bias=True)

        # Define optimizer, loss
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        accuracy_fn = f1_score

        # Stratified

        shuffled_indices = list(RandomSampler(dataset))
        val_iter = (shuffled_indices[round(i * len(dataset) / k_folds): round((i+1) * len(dataset) / k_folds)] for i in range(k_folds))
        
        val_indices = next(val_iter)
        train_indices = list(set(shuffled_indices) - set(val_indices))

        train_set = Subset(dataset, train_indices)
        val_set = Subset(dataset, val_indices)
        
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True) # num_workers = ?
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

        # Initial check
        # untrained = test(val_loader, model, loss_fn, accuracy_fn)
        # print(f'Untrained model should score ~ 50 % ? The model is scoring {untrained[1] * 100:.2f} % .\nIf not then I haven\'t stratified the data?')

        # The loop
        train_accuracy_per_epoch = []
        val_accuracy_per_epoch = []
        for e in range(EPOCHS):
            _, train_acc = train(train_loader, model, loss_fn, optimizer, accuracy_fn)
            train_accuracy_per_epoch.append(train_acc.item())
            _, val_acc = test(val_loader, model, loss_fn, accuracy_fn) # changed from test=valid then using valid to just test
            val_accuracy_per_epoch.append(val_acc.item())
        
            print(f'Fold {i+1} | Epoch {e+1} | Train acc {train_acc.item() * 100:.4f} % | Val acc {val_acc.item() * 100:.4f} %')

        train_accuracies.append(max(train_accuracy_per_epoch)) # max or mean
        val_accuracies.append(max(val_accuracy_per_epoch))
        print(f'Fold {i+1} | MAX | Train acc {train_accuracies[-1] * 100:.4f} % | Val acc {val_accuracies[-1] * 100:.4f} %')

    train_accuracy_across_folds = np.mean(train_accuracies)
    val_accuracy_across_folds = np.mean(val_accuracies)
    print(f'MEAN ACROSS FOLDS | Train acc {train_accuracy_across_folds * 100:.4f} % | Val acc {val_accuracy_across_folds * 100:.4f} %')
    
    # return (train_accuracy_across_folds, val_accuracy_across_folds)
    return train_accuracies, val_accuracies # should be (5,) each


# --------------------------------------------------------------------------------------------------

def run(model, weights, dataset):
    
    train_accuracies, val_accuracies = cross_validate(in_model=model, in_weights=weights, dataset=dataset, k_folds=KFOLDS)

    return train_accuracies, val_accuracies

