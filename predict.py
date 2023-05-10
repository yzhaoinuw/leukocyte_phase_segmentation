# -*- coding: utf-8 -*-
"""
Created on Tue May  9 11:03:50 2023

@author: Yue
"""

from tqdm import tqdm
import torch
from torch import nn, autocast
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchmetrics import ConfusionMatrix, Dice
from sklearn.model_selection import KFold

from utils import show_image
from unet import UNet
from dataset import SegmentationDataset, Subset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "./data/test"
MODEL_PATH = "./model/"
model_name = "model_CELoss_weighted_30"
transform_fn = transforms.Compose(
    [transforms.ToTensor(),
    ]
)

test_data = SegmentationDataset(DATA_PATH, transform=transform_fn)
test_dataloader = DataLoader(test_data, batch_size=16)

loss_fn = nn.CrossEntropyLoss()
confmat = ConfusionMatrix(task="multiclass", num_classes=4)
dice = Dice(num_classes=4)
model = UNet(padding=True, up_mode='upsample')
model.load_state_dict(torch.load(MODEL_PATH + model_name))
model.eval()

test_loss = 0
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
        dice.update(pred_masks, masks)
        
test_loss = test_loss/batch
print(f"test loss: {loss:.3f}")

#%%
sample, mask = test_data[-1]
output = model(sample.unsqueeze(0))
pred_masks = F.softmax(output, dim=1)
pred_masks = torch.argmax(pred_masks, dim=1)
show_image(pred_masks.squeeze())
show_image(mask)
