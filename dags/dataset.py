import json
import os

import torch
from PIL import Image


### Dataset class
class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None, target_transform=None, images_list=None):
        self.path = path
        self.transform = transform
        self.target_transform = target_transform
        if images_list is not None:
            self.images = sorted(list(images_list))
        else:
            self.images = list(sorted(os.listdir(self.path['images'])))
        with open(self.path['targets'], 'r') as file:
            self.targets = json.load(file)

    def __getitem__(self, index):
        path_image = os.path.join(self.path['images'], self.images[index])
        image = Image.open(path_image)

        target = {}
        target['boxes'] = torch.as_tensor(
            self.targets[self.images[index]]['boxes'], dtype=torch.float32
        )
        target['labels'] = torch.as_tensor(
            self.targets[self.images[index]]['labels'], dtype=torch.int64
        )
        target['image_id'] = torch.tensor([index])
        target['area'] = (target['boxes'][:, 3] - target['boxes'][:, 1]) * (
            target['boxes'][:, 2] - target['boxes'][:, 0]
        )
        target['iscrowded'] = torch.zeros_like(
            target['labels'], dtype=torch.int64
        )  # not crowded

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.images)
