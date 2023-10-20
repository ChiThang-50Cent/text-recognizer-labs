import torch
import torchvision
import random
import numpy as np

from torchvision import transforms, datasets
from torch.utils.data import random_split

def get_EMNIST_datasets():
    train_set = datasets.EMNIST(root='./', split='balanced', download=True, train=True, 
                                transform=transforms.Compose([transforms.ToTensor()]))
    test_set = datasets.EMNIST(root='./', split='balanced', download=True, train=False, 
                                transform=transforms.Compose([transforms.ToTensor()]))
    
    train_size = int(0.9 * len(train_set))
    valid_size = len(train_set) - train_size

    train_set, valid_set = random_split(train_set, [train_size, valid_size])

    print(f'train_size: {train_size}, valid_size: {valid_size}, test_size: {len(test_set)}')

    return train_set, valid_set, test_set

def set_all_seed(seed_val):
  random.seed(seed_val)
  np.random.seed(seed_val)
  torch.manual_seed(seed_val)
  torch.cuda.manual_seed_all(seed_val)

def flat_acc(y_hat , y_true):
  test = y_hat.softmax(dim=1).argmax(dim=1).numpy()
  y_test = y_true.numpy()

  return np.sum(test==y_test) / len(test)

def get_device():
  if torch.cuda.is_available():
    device = torch.device("cuda")
  else:
    device = torch.device("cpu")
  return device

def training_loop(n_epochs, model, train_loader, val_loader, loss_fn, optimizer, scheduler, device):
  
  for epoch in range(n_epochs):

    total_train_loss = 0
    model.train()

    print(f'Epoch {epoch + 1} in {n_epochs} total.', end=' ')

    for batch in train_loader:
      
      batch = (t.to(device) for t in batch)
      X, y = batch

      optimizer.zero_grad()

      y_hat = model(X)
      loss = loss_fn(y_hat, y)
      loss.backward()

      optimizer.step()
      scheduler.step()

      total_train_loss += loss.item()

    print(f'Train loss {total_train_loss/len(train_loader):.2f}.', end=' ')

    total_val_loss = 0
    total_val_acc = 0
    with torch.no_grad():
      model.eval()
      for batch in val_loader:
        batch = (t.to(device) for t in batch)
        X, y = batch

        y_hat = model(X)
        loss = loss_fn(y_hat, y)

        y_hat = y_hat.detach().cpu()
        y = y.to('cpu')

        total_val_loss += loss.item()
        total_val_acc += flat_acc(y_hat, y)

    print(f'Valid loss {total_val_loss/len(val_loader):.2f}.', end=' ')
    print(f'Valid acc {total_val_acc/len(val_loader):.2f}.')