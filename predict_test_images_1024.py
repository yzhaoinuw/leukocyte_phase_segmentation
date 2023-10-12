# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 18:22:30 2023

@author: Yue
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt

import cv2

import torch

# from torch import nn
import torch.nn.functional as F

# from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchvision.transforms as transforms


from utils import show_image
from unet import UNet
from config import mask_mapping_reversed

# from dataset import SegmentationDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "./test_images/"
FOLDER_NAME = "PC4"
SPLIT_NUM = 1
MODEL_PATH = f"./model_large_upconv_gaussianBlur_random/split_{SPLIT_NUM}/"
SAVE_PATH = os.path.join(DATA_PATH, FOLDER_NAME + "_pred")
MODEL_NAME = "model_CELoss_weighted_51.pth"

Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)

channels, height, width = 1, 1024, 1024
patch_size = 256
stride = patch_size  # No overlap

transform_fn = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

model = UNet(padding=True, up_mode="upconv")
model.load_state_dict(torch.load(MODEL_PATH + MODEL_NAME))
model.eval()

# %%
folder_path = os.path.join(DATA_PATH, FOLDER_NAME)
img_folder = os.listdir(folder_path)
for img_file in img_folder:
    if not img_file.endswith(".tif"):
        continue
    print(img_file)
    img_name = img_file.split(".tif")[0]
    img_path = os.path.join(folder_path, img_file)
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    image = transform_fn(image)
    # Sliding window parameters
    patches_row = (height - patch_size) // stride + 1
    patches_col = (width - patch_size) // stride + 1
    # Extract patches
    patches = image.unfold(1, patch_size, stride).unfold(2, patch_size, stride)
    patches = patches.permute(1, 2, 0, 3, 4).reshape(
        -1, channels, patch_size, patch_size
    )

    outputs = model(patches)
    pred_masks = F.softmax(outputs, dim=1)
    pred_masks = torch.argmax(pred_masks, dim=1, keepdim=True)
    # Combine output patches to get the full image
    combined_mask = torch.zeros(channels, height, width)
    for i in range(patches_row):
        for j in range(patches_col):
            combined_mask[
                :,
                i * stride : i * stride + patch_size,
                j * stride : j * stride + patch_size,
            ] = pred_masks[i * patches_col + j]

    for k in mask_mapping_reversed:
        combined_mask[combined_mask == k] = mask_mapping_reversed[k]
    save_masks = combined_mask.float() / 255
    save_image(save_masks, os.path.join(SAVE_PATH, img_name + ".tif"))
