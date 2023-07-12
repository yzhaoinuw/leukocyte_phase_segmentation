# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 22:07:10 2023

@author: Yue
"""

import logging
from pathlib import Path

from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchmetrics import ConfusionMatrix
from sklearn.model_selection import KFold

from utils import calculate_dice, calculate_accuracy
from unet import UNet
from dataset import SegmentationDataset, Subset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "./data/train"
CLASS_NUM = 4
N_SPLIT = 4
TRAIN_BS = 16
VAL_BS = 16
MODEL_PATH = f"./model_upconv/split_{N_SPLIT}/"
Path(MODEL_PATH).mkdir(parents=True, exist_ok=True)

transform_fn = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

data = SegmentationDataset(DATA_PATH, transform=transform_fn)
data_size = len(data)
kf = KFold(n_splits=5)
splits = kf.split(list(range(data_size)))
train_indices, val_indices = list(splits)[N_SPLIT]
train_data = Subset(data, train_indices)
val_data = Subset(data, val_indices)
train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=16)
class_count = torch.zeros(CLASS_NUM)
for batch, (images, masks) in enumerate(train_dataloader, 1):
    class_count += masks.unique(return_counts=True)[1]

epochs = 100
model = UNet(padding=True, up_mode="upconv").to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    patience=len(val_data) // VAL_BS * 5,
    factor=0.5,
)
loss_weight = torch.Tensor([1, 1, 1, 1]).to(device)
loss_fn = nn.CrossEntropyLoss(weight=loss_weight)
# loss_fn = dice_loss
best_val_loss = 1000
scaler = torch.cuda.amp.GradScaler()

logger = logging.getLogger("logger")
logging.getLogger().setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s: %(message)s")
file_handler = logging.FileHandler(
    filename=MODEL_PATH + f"train_CELoss_{loss_weight.cpu().numpy()}.log", mode="w"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logging.getLogger("logger").info(f"Training on split {N_SPLIT} \n")
logging.getLogger("logger").info(f"Phase class count: {class_count}")
logging.getLogger("logger").info(
    f"Phase class count: {class_count / torch.sum(class_count)} \n"
)
logging.getLogger("logger").info(f"CE loss weight: {loss_weight}")

best_model = model
# %%
for epoch in range(1, epochs + 1):
    model.train()
    epoch_loss = 0
    val_loss = 0
    confmat = ConfusionMatrix(task="multiclass", num_classes=4).to(device)
    with tqdm(
        total=len(train_data), unit=" batch", desc=f"Epoch {epoch}/{epochs}"
    ) as pbar:
        for batch, (images, masks) in enumerate(train_dataloader, 1):
            images, masks = images.to(device), masks.to(device, dtype=torch.long)

            # Runs the forward pass with autocasting.
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = loss_fn(output, masks)
                # loss += dice_loss(output, masks)

            # Backpropagation
            optimizer.zero_grad()
            # Scales the loss, and calls backward() on the scaled loss to create scaled gradients.
            scaler.scale(loss).backward()
            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
            scaler.step(optimizer)
            # Updates the scale for next iteration.
            scaler.update()

            # loss.backward()
            # optimizer.step()

            pbar.update(images.shape[0])
            epoch_loss += loss.item()
        epoch_loss /= batch
        pbar.set_postfix({"Batch": batch, "Train loss (in progress)": loss})

        # eval
        model.eval()
        with torch.no_grad():
            for batch, (images, masks) in enumerate(val_dataloader, 1):
                images, masks = images.to(device), masks.to(device, dtype=torch.long)
                output = model(images)
                loss = loss_fn(output, masks)
                scheduler.step(loss)
                val_loss += loss.item()
                pred_masks = F.softmax(output, dim=1)
                pred_masks = torch.argmax(pred_masks, dim=1)
                confmat.update(pred_masks, masks)

        dice = calculate_dice(confmat.confmat)
        accuracy = calculate_accuracy(confmat.confmat)
        val_loss /= batch
        pbar.set_postfix(
            {"Train loss (epoch)": epoch_loss, "Validation loss": val_loss}
        )
        logging.getLogger("logger").info(f"Epoch {epoch}")
        logging.getLogger("logger").info(f"Learning rate: {scheduler._last_lr[0]}")
        logging.getLogger("logger").info(f"Train Loss: {epoch_loss:.3f}")
        logging.getLogger("logger").info(f"Validation Loss: {val_loss:.3f}")
        logging.getLogger("logger").info(f"Accuracy: {accuracy:.3f}")
        logging.getLogger("logger").info(f"Dice Coefficients: {dice}")
        logging.getLogger("logger").info("\n")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            model_name = "model_{}_{}.pth".format("CELoss_weighted", epoch)
        if scheduler._last_lr[0] < 0.00001:
            logging.getLogger("logger").info(
                "Applying early stopping. Training Terminated."
            )
            break

logging.getLogger("logger").info(f"Confusion matrix:\n{confmat.confmat}")
torch.save(best_model.state_dict(), MODEL_PATH + model_name)
handlers = logger.handlers
for handler in handlers:
    logger.removeHandler(handler)
    handler.close()
