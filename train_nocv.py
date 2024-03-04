import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.dataset import Subset
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

def train(dataloader, model, loss_fn, optimizer, accuracy_fn, device):
    
    n_samples = 0
    train_loss = 0
    train_acc = 0
    model = model.to(device)

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y = y.long()

        pred = model(X).float()
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_acc = accuracy_fn(y.int().cpu().numpy(), pred.argmax(1).cpu().numpy()).item()

        n_samples += X.size(0)
        train_loss += loss.item() * X.size(0)
        train_acc += batch_acc * X.size(0)

        if batch % int(len(dataloader) / 10) == 0:
            print(f'Batch {batch} of {len(dataloader)}: Train loss = {loss.item()}')

    train_loss /= n_samples
    train_acc /= n_samples

    print('Epoch train loss:', train_loss)
    print('Train acc:', train_acc)

    return train_loss, train_acc, model

def test(dataloader, model, accuracy_fn, device):
    
    n_samples = 0
    test_acc = 0
    model = model.to(device)

    model.eval()
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y = y.long()

            pred = model(X).float()
            batch_acc = accuracy_fn(y.int().cpu().numpy(), pred.argmax(1).cpu().numpy()).item()

            n_samples += X.size(0)
            test_acc += batch_acc * X.size(0)

        test_acc /= n_samples
    
    return test_acc

def run(input_model, input_weights, train_dataset, val_dataset, test_dataset, epochs=10, lr=1e-4, patience=5):

    OUT_FEATURES = 2
    SEED = 1999
    VAL_SIZE = 0.2
    BATCH_SIZE = 32
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('Using:', device)
    print('Learning rate:', lr)
    print('Patience:', patience)

    model = input_model(weights=input_weights)
    model = model.to(device)
    
    # Change last layer to binary
    if model.__class__.__name__ in ['ConvNeXt', 'EfficientNet', 'VGG']:
        model.classifier[-1] = nn.Linear(in_features=model.classifier[-1].in_features, out_features=OUT_FEATURES, bias=True)
    elif model.__class__.__name__ in ['ResNet', 'Inception3']:
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=OUT_FEATURES, bias=True)
    elif model.__class__.__name__ in ['VisionTransformer']:
        model.heads[-1] = nn.Linear(in_features=model.heads[-1].in_features, out_features=OUT_FEATURES, bias=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    accuracy_fn = f1_score

    # Train-val split

    # shuffled_indices = list(RandomSampler(train_dataset))

    # train_idx = shuffled_indices[:int((1-VAL_SIZE)*len(train_dataset))]
    # val_idx = shuffled_indices[int((1-VAL_SIZE)*len(train_dataset)):]

    # train_set = Subset(train_dataset, train_idx)
    # val_set = Subset(train_dataset, val_idx)
    
    # train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True) # num_workers = ?
    # val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # The loop
    train_accuracy_per_epoch = []
    val_accuracy_per_epoch = []

    best_val_acc = 0
    best_model = None

    no_improvement = 0
    for e in range(epochs):
        print('*'*50)
        print(f'Starting Epoch {e+1}')
        _, train_acc, train_model = train(dataloader=train_loader, model=model, loss_fn=loss_fn, optimizer=optimizer, accuracy_fn=accuracy_fn, device=device)
        train_accuracy_per_epoch.append(train_acc)
        val_acc = test(dataloader=val_loader, model=model,accuracy_fn=accuracy_fn, device=device)
        print(f'Val acc: {val_acc * 100:.4f} %')
        val_accuracy_per_epoch.append(val_acc)

        print(f'Epoch {e+1} | Train acc {train_acc * 100:.4f} % | Val acc {val_acc * 100:.4f} %')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = train_model
            print('New model saved.')
            no_improvement = 0
        else:
            no_improvement += 1
            print(f'Patience {no_improvement}/{patience}')

        if no_improvement == patience:
            print('Stopped early.')
            break
    

    # Test
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    with torch.inference_mode():
        test_acc = test(dataloader=test_loader, model=best_model, accuracy_fn=accuracy_fn, device=device)

        print(f'Test acc: {test_acc * 100:.4f} %')
        print('\n', '='*50)

    return train_accuracy_per_epoch, val_accuracy_per_epoch, test_acc





# --------------------------------------------------------------------------------------------------








