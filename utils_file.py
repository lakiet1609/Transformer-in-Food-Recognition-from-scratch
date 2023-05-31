import torch
from pathlib import Path
import matplotlib.pyplot as plt 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#Plot the comparision between the test set and train set  
def plot_evaluation_curve(results):
    train_loss = torch.tensor(results['train_loss']).to('cpu')
    test_loss = torch.tensor(results['test_loss']).to('cpu')
    train_accuracy = torch.tensor(results["train_acc"]).to('cpu')
    test_accuracy = torch.tensor(results["test_acc"]).to('cpu')
    epochs = range(len(results['train_loss']))
    plt.figure(figsize=(15, 7))
    
    # Plot loss
    plt.plot(epochs, train_loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()
    
    #Plot accuracy
    plt.plot(epochs, train_accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()

#Save model
def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    torch.save(obj=model.state_dict(),
             f=model_save_path)