from __future__ import print_function
from PIL import Image
import glob
import torch.utils.data as data
from tqdm.notebook import trange, tqdm
import random
from torchvision.transforms import Compose, ColorJitter, RandomHorizontalFlip

class ImageNetLimited(data.Dataset):
    """ImageNet Limited dataset."""
    
    def __init__(self, root_dir, transform=None):
        
        if transform:
            self.transform = transform

        flatten = lambda l: [item for sublist in l for item in sublist]
        files = flatten([glob.glob(e) for e in [root_dir+'/[0-9]/*.jpg',root_dir+'/[0-9][0-9]/*.jpg']])

        self.images=[]
        self.labels=[]
        
        for file in tqdm(files):
            temp = Image.open(file)
            keep = temp.copy()
            self.labels.append(int(file.split('/')[-2]))
            self.images.append(keep)
            temp.close()

    def __len__(self):
    
        return len(self.images)

    def __getitem__(self, idx):

        label = self.labels[idx]
        image = self.images[idx]

        if self.transform:
            image = self.transform(image)

        return image,label
