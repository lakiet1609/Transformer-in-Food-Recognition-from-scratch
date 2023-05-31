from torch.utils.data import Dataset
import pathlib
import os
from PIL import Image
from typing import List
from torchvision import transforms


def find_class(directory):
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f'Could not find any classes in {directory}')
    class_to_idx = {class_name: i for i,class_name in enumerate(classes)}
    return classes, class_to_idx

def data_augmentation(image_size: tuple):
    data_transform  = transforms.Compose([
        transforms.Resize(size=image_size),
        transforms.ToTensor()
    ])
    return data_transform

#create custom dataset
class imagefolder_custom(Dataset):
    def __init__(self, target_directory, transform = None):
        #Target directory in this case like train or test folder
        self.paths = list(pathlib.Path(target_directory).glob('*/*.jpg'))
        self.transform = transform
        self.classes, self.class_to_idx = find_class(target_directory)
    
    def load_image(self,idx: int):
        image_path = self.paths[idx]
        return Image.open(image_path)
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx: int):
        img= self.load_image(idx=idx)
        class_name =  self.paths[idx].parent.name
        class_idx = self.class_to_idx[class_name]
        if self.transform:
            return self.transform(img), class_idx 
        else:
            return img, class_idx

