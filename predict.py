# -*- coding: utf-8 -*-
"""
Created on Tue May  9 11:03:50 2023

@author: Yue
"""

import os
import glob
import logging

from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchmetrics import ConfusionMatrix

from utils import show_image, calculate_dice, calculate_accuracy
from unet import UNet
from dataset import SegmentationDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "./data_large/test"

SPLIT_NUM = 4
CLASS_NUM = 4
BATCH_SIZE = 16
MODEL_PATH = f"./model_large_upconv_gaussian_sobel_random0.8/split_{SPLIT_NUM}/"
model_path = glob.glob(os.path.join(MODEL_PATH, "*.pth"))[0]

transform_fn = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

test_data = SegmentationDataset(DATA_PATH, transform=transform_fn)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

class_count = torch.zeros(CLASS_NUM)
for batch, (images, masks) in enumerate(test_dataloader, 1):
    class_count += masks.unique(return_counts=True)[1]

logger = logging.getLogger("logger")
logging.getLogger().setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s: %(message)s")
file_handler = logging.FileHandler(filename=MODEL_PATH + "test.log", mode="w")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logging.getLogger("logger").info(f"Model name: {model_path}")
logging.getLogger("logger").info(f"Phase class count: {class_count}")
logging.getLogger("logger").info(
    f"Phase class count: {class_count / torch.sum(class_count)} \n"
)

loss_fn = nn.CrossEntropyLoss()
confmat = ConfusionMatrix(task="multiclass", num_classes=4)
model = UNet(padding=True, up_mode="upconv")
model.load_state_dict(torch.load(model_path))
model.eval()

test_loss = 0
with tqdm(total=len(test_data), unit=" images") as pbar:
    with torch.no_grad():
        for batch, (images, masks) in enumerate(test_dataloader, 1):
            images, masks = images, masks.long()

            # Compute prediction and loss
            output = model(images)
            loss = loss_fn(output, masks)
            test_loss += loss.item()
            pred_masks = F.softmax(output, dim=1)
            pred_masks = torch.argmax(pred_masks, dim=1)
            confmat.update(pred_masks, masks)
            pbar.update(BATCH_SIZE)
    pbar.set_postfix({"Batch": batch, "test loss (in progress)": loss})

dice = calculate_dice(confmat.confmat)
accuracy = calculate_accuracy(confmat.confmat)
test_loss = test_loss / batch
logging.getLogger("logger").info(f"Test Loss: {test_loss:.3f}")
logging.getLogger("logger").info(f"Accuracy: {accuracy:.3f}")
logging.getLogger("logger").info(f"Dice Coefficients: {dice}")
logging.getLogger("logger").info("\n")
logging.getLogger("logger").info(f"Confusion matrix:\n{confmat.confmat}")
handlers = logger.handlers
for handler in handlers:
    logger.removeHandler(handler)
    handler.close()

# %%

sample, mask = test_data[5]
output = model(sample.unsqueeze(0))
pred_masks = F.softmax(output, dim=1)
pred_masks = torch.argmax(pred_masks, dim=1)
show_image(pred_masks.squeeze())
show_image(mask)
