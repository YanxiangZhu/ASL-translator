import numpy as np
from skimage import io
from skimage import transform
import matplotlib.pyplot as plt
import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.utils import make_grid

class ZSLdataset(Dataset):

    def __init__(self, root_dir, names_file, transform=None):
        self.root_dir = root_dir
        self.names_file = names_file
        self.transform = transform
        self.size = 0
        self.names_list = []

        if not os.path.isfile(self.names_file):
            print(self.names_file + 'does not exist!')
        file = open(self.names_file)
        for f in file:
            self.names_list.append(f)
            self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image_path = self.root_dir + self.names_list[idx].split(' ')[0]
        if not os.path.isfile(image_path):
            print(image_path + 'does not exist!')
            return None
        image = io.imread(image_path)
        label = int(self.names_list[idx].split(' ')[1])

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        return sample


train_dataset = ZSLdataset(root_dir='./data/asl_alphabet_train',
                          names_file='./data/asl_alphabet_train/train.txt',
                          transform=None)

plt.figure()
for (cnt,i) in enumerate(train_dataset):
    image = i['image']
    label = i['label']

    ax = plt.subplot(4, 4, cnt+1)
    ax.axis('off')
    ax.imshow(image)
    ax.set_title('label {}'.format(label))
    plt.pause(0.001)

    if cnt == 15:
        break
