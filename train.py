# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 22:07:10 2023

@author: Yue
"""

import logging 

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
DATA_PATH = "./data/train"
MODEL_PATH = "./model/"
transform_fn = transforms.Compose(
    [transforms.ToTensor(),
    ]
)

data = SegmentationDataset(DATA_PATH, transform=transform_fn)
data_size = len(data)
kf = KFold(n_splits=5)
splits = kf.split(list(range(data_size)))
train_indices, val_indices = list(splits)[-1]
train_data = Subset(data, train_indices)
val_data = Subset(data, val_indices)
train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=16)
#train_features, train_labels = next(iter(train_dataloader))

epochs = 30
model = UNet(padding=True, up_mode='upsample').to(device)
optimizer = torch.optim.Adam(model.parameters())
loss_weight = torch.Tensor([1, 5, 5, 5]).to(device)
loss_fn = nn.CrossEntropyLoss(weight=loss_weight)
best_val_loss = 1000

logging.basicConfig(
    filename=MODEL_PATH+"train.log",
    filemode='a',
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG
)
#%%
for epoch in range(1, epochs + 1):
    model.train()
    epoch_loss = 0
    with tqdm(total=len(train_data), desc=f'Epoch {epoch}/{epochs}') as pbar:
        for batch, (images, masks) in enumerate(train_dataloader, 1):
            images, masks = images.to(device), masks.to(device, dtype=torch.long)

            # Compute prediction and loss
            output = model(images)
            loss = loss_fn(output, masks)
        
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.update(images.shape[0])
            epoch_loss += loss.item()
        pbar.set_postfix(**{'Train Loss': epoch_loss/batch})
        #print(loss.item())
        
        model.eval()
        val_loss = 0
        confmat = ConfusionMatrix(task="multiclass", num_classes=4).to(device)
        dice = Dice(num_classes=4).to(device)
        with torch.no_grad():
            for batch, (images, masks) in enumerate(val_dataloader, 1):
                images, masks = images.to(device), masks.to(device, dtype=torch.long)

                # Compute prediction and loss
                # Compute prediction and loss
                output = model(images)
                loss = loss_fn(output, masks)
                val_loss += loss.item()
                pred_masks = F.softmax(output, dim=1)
                pred_masks = torch.argmax(pred_masks, dim=1)
                confmat.update(pred_masks, masks)
                dice.update(pred_masks, masks)

        val_loss = val_loss/batch
        logging.info(f"Epoch {epoch}")
        logging.info(f"Validation Loss: {val_loss:.3f}")
        logging.info(f"Dice Score: {dice.compute()}")
        logging.info("\n")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            model_name = 'model_{}_{}'.format('CELoss_weighted', epoch)
torch.save(best_model.state_dict(), MODEL_PATH + model_name)
        