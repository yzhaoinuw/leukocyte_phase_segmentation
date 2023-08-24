# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 15:49:16 2023

@author: Yue
"""

import os
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
DATA_PATH = "./test_images/256/Section2"
SPLIT_NUM = 1
MODEL_PATH = f"./model_large_upconv_gaussianBlur_random/split_{SPLIT_NUM}/"
SAVE_PATH = "./test_images/256/Section2_masks"
model_name = "model_CELoss_weighted_51.pth"
transform_fn = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

model = UNet(padding=True, up_mode="upconv")
model.load_state_dict(torch.load(MODEL_PATH + model_name))
model.eval()

# %%
img_folder = os.listdir(DATA_PATH)
for img_file in img_folder:
    if not img_file.endswith(".tif"):
        continue
    print(img_file)
    img_name = img_file.split(".tif")[0]
    img_path = os.path.join(DATA_PATH, img_file)
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    image = transform_fn(image)
    output = model(image.unsqueeze(0))
    pred_masks = F.softmax(output, dim=1)
    pred_masks = torch.argmax(pred_masks, dim=1)
    for k in mask_mapping_reversed:
        pred_masks[pred_masks == k] = mask_mapping_reversed[k]
    # show_image(image.squeeze())
    # show_image(pred_masks.squeeze())
    save_masks = pred_masks.float() / 255
    save_image(save_masks, os.path.join(SAVE_PATH, img_name + ".tif"))
    # print("\n")
