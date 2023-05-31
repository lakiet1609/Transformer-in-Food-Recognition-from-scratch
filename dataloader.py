import os
import custom_dataset
from custom_dataset import data_augmentation
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str, 
    test_dir: str, 
    transform: data_augmentation, 
    batch_size: int, 
    num_workers: int=NUM_WORKERS):

  # Use ImageFolder to create dataset(s)
  train_data = custom_dataset.imagefolder_custom(train_dir, transform=transform)
  test_data = custom_dataset.imagefolder_custom(test_dir, transform=transform)

  # Get class names
  class_names = train_data.classes

  # Turn images into data loaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
  )

  return train_dataloader, test_dataloader, class_names