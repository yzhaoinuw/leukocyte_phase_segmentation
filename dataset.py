# -*- coding: utf-8 -*-
"""
Created on Sun May  7 17:37:38 2023

@author: Yue
based on https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
"""

import re
import os

import cv2

# from torch import Tensor
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(self, data_path, indices=[], transform=None, target_transform=None):
        super(SegmentationDataset, self).__init__()
        image_path = os.path.join(data_path, "images")
        mask_path = os.path.join(data_path, "masks")
        image_files = {}
        mask_files = {}
        for file in os.listdir(image_path):
            index = re.search("[0-9]+", file)
            if index is None:
                continue
            index = int(index.group())
            if indices and index not in indices:
                continue
            image_files[index] = os.path.join(image_path, file)

        for file in os.listdir(mask_path):
            index = re.search("[0-9]+", file)
            if index is None:
                continue
            index = int(index.group())
            if indices and index not in indices:
                continue
            mask_files[index] = os.path.join(mask_path, file)

        self.transform = transform
        self.target_transform = target_transform
        # self.mapping = {209: 0, 188: 1, 157: 2, 76: 3}
        self.mapping = {209: 0, 188: 3, 157: 2, 76: 1}

        assert image_files.keys() == mask_files.keys()
        self.image_files = [
            image_path for ind, image_path in sorted(image_files.items())
        ]
        self.mask_files = [mask_path for ind, mask_path in sorted(mask_files.items())]

    def mask_to_class(self, mask):
        for k in self.mapping:
            mask[mask == k] = self.mapping[k]
        return mask

    def __getitem__(self, index):
        img_path = self.image_files[index]
        mask_path = self.mask_files[index]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        mask = self.mask_to_class(mask)
        return image, mask

    def __len__(self):
        return len(self.image_files)
