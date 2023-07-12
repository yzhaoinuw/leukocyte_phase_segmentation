# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 15:09:56 2023

@author: Yue
"""

import time
import copy
import logging

from tqdm import tqdm

import torch
from torch import nn, vmap
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.func import stack_module_state, functional_call
import torchvision.transforms as transforms

from torchmetrics import ConfusionMatrix

from utils import show_image, calculate_dice, calculate_accuracy
from unet import UNet
from dataset import SegmentationDataset


start_time = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "./data/test"
CLASS_NUM = 4
MODEL_PATH = "./model_upconv/"
""" augmented
model_names = [#"split_0/model_CELoss_weighted_20.pth",
               #"split_1/model_CELoss_weighted_24.pth",
               "split_2/model_CELoss_weighted_30.pth",
               "split_3/model_CELoss_weighted_28.pth",
               #"split_4/model_CELoss_weighted_34.pth",
               ]
"""
model_names = [
    "split_0/model_CELoss_weighted_31.pth",
    "split_1/model_CELoss_weighted_53.pth",
    "split_2/model_CELoss_weighted_36.pth",
    "split_3/model_CELoss_weighted_45.pth",
    "split_4/model_CELoss_weighted_36.pth",
]
transform_fn = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

test_data = SegmentationDataset(DATA_PATH, transform=transform_fn)
test_dataloader = DataLoader(test_data, batch_size=16)
class_count = torch.zeros(CLASS_NUM)
for batch, (images, masks) in enumerate(test_dataloader, 1):
    class_count += masks.unique(return_counts=True)[1]

logger = logging.getLogger("logger")
logging.getLogger().setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s: %(message)s")
file_handler = logging.FileHandler(filename=MODEL_PATH + "test_ensemble.log", mode="w")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logging.getLogger("logger").info(f"Phase class count: {class_count}")
logging.getLogger("logger").info(
    f"Phase class count: {class_count / torch.sum(class_count)} \n"
)

loss_fn = nn.CrossEntropyLoss()
confmat = ConfusionMatrix(task="multiclass", num_classes=4)
models = []

time1 = time.time()
prep_time = time1 - start_time
print(f"prep_time: {prep_time} seconds")

for i in range(len(model_names)):
    model = UNet(padding=True, up_mode="upconv")
    model.load_state_dict(torch.load(MODEL_PATH + model_names[i]))
    model.eval()
    models.append(model)

params, buffers = stack_module_state(models)
time2 = time.time()
load_time = time2 - start_time
print(f"load_time: {load_time} seconds")

# Construct a "stateless" version of one of the models. It is "stateless" in
# the sense that the parameters are meta Tensors and do not have storage.
base_model = copy.deepcopy(models[0])
base_model = base_model.to("meta")


def fmodel(params, buffers, x):
    return functional_call(base_model, (params, buffers), (x,))


test_data = SegmentationDataset(DATA_PATH, transform=transform_fn)
test_dataloader = DataLoader(test_data, batch_size=16)


test_loss = 0
with tqdm(total=len(test_data), unit=" batch") as pbar:
    with torch.no_grad():
        for batch, (images, masks) in enumerate(test_dataloader, 1):
            images, masks = images, masks.long()

            # time3 = time.time()
            # ensemble_time = time3 - start_time
            # print(f"ensemble_time: {ensemble_time} seconds")

            output = vmap(fmodel, in_dims=(0, 0, None))(params, buffers, images)

            # time4 = time.time()
            # pred_time = time4 - start_time
            # print(f"pred_time: {pred_time} seconds")

            output_ensemble = torch.mean(output, dim=0)
            loss = loss_fn(output_ensemble, masks)
            test_loss += loss.item()
            pred_masks = F.softmax(output_ensemble, dim=1)
            pred_masks = torch.argmax(pred_masks, dim=1)
            confmat.update(pred_masks, masks)
            pbar.update(images.shape[0])

    pbar.set_postfix({"Batch": batch, "test loss (in progress)": loss})

test_loss = test_loss / batch
dice = calculate_dice(confmat.confmat)
accuracy = calculate_accuracy(confmat.confmat)
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
"""
sample, mask = test_data[189]
output = vmap(fmodel, in_dims=(0, 0, None))(params, buffers, sample.unsqueeze(0))
output_ensemble = torch.mean(output, dim=0)
pred_masks = F.softmax(output_ensemble, dim=1)
pred_masks = torch.argmax(pred_masks, dim=1)
show_image(pred_masks.squeeze())
show_image(mask)
"""
