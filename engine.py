import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#Create train and test loops function
def train_step(model: nn.Module,
               dataloader: DataLoader,
               loss_function: nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy,
               device = device):
    model.to(device)
    model.train()
    train_loss, train_acc = 0, 0
    for batch, (X,y) in enumerate(dataloader):
        X,y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_function(y_pred,y)
        train_loss += loss.item()
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += accuracy(y_pred_class,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc

def test_step(model: nn.Module,
               dataloader: DataLoader,
               loss_function: nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy,
               device = device):
    model.to(device)
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for batch, (X,y) in enumerate(dataloader):
            X,y = X.to(device), y.to(device)
            test_pred = model(X)
            loss = loss_function(test_pred,y)
            test_loss += loss.item()
            test_pred_class = torch.argmax(torch.softmax(test_pred, dim=1), dim=1)
            test_acc += accuracy(test_pred_class,y)
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
    return test_loss, test_acc
   
#Create function for training process 
def training(model: torch.nn.Module, 
          train_dataloader: DataLoader, 
          test_dataloader: DataLoader, 
          optimizer: torch.optim.Optimizer,
          accuracy,
          loss_function: nn.Module,
          epochs: int,
          device = device):
    
    results = {"train_loss": [],
                "train_acc": [],
                "test_loss": [],
                "test_acc": []
                }
    
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_function=loss_function,
                                           accuracy=accuracy,
                                           optimizer=optimizer,
                                           device=device)
        
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_function=loss_function,
                                        optimizer=optimizer,
                                        accuracy=accuracy,
                                        device=device)
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )
        
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        
    return results
