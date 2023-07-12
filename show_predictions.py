# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 14:04:02 2023

@author: Yue
"""

import copy

import torch
from torch import vmap
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.func import stack_module_state, functional_call
import torchvision.transforms as transforms

from utils import show_image
from unet import UNet
from dataset import SegmentationDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "./data_augmented/test"
CLASS_NUM = 4
MODEL_PATH = "./model_augmented_upconv/"

model_names = [  # "split_0/model_CELoss_weighted_20.pth",
    # "split_1/model_CELoss_weighted_24.pth",
    "split_2/model_CELoss_weighted_30.pth",
    "split_3/model_CELoss_weighted_28.pth",
    # "split_4/model_CELoss_weighted_34.pth",
]
transform_fn = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

test_data = SegmentationDataset(DATA_PATH, transform=transform_fn)
test_dataloader = DataLoader(test_data, batch_size=16)

models = []
for i in range(len(model_names)):
    model = UNet(padding=True, up_mode="upconv")
    model.load_state_dict(torch.load(MODEL_PATH + model_names[i]))
    model.eval()
    models.append(model)

params, buffers = stack_module_state(models)


# Construct a "stateless" version of one of the models. It is "stateless" in
# the sense that the parameters are meta Tensors and do not have storage.
base_model = copy.deepcopy(models[0])
base_model = base_model.to("meta")


def fmodel(params, buffers, x):
    return functional_call(base_model, (params, buffers), (x,))


test_data = SegmentationDataset(DATA_PATH, transform=transform_fn)
test_dataloader = DataLoader(test_data, batch_size=16)

# %%
sample, mask = test_data[189]
output = vmap(fmodel, in_dims=(0, 0, None))(params, buffers, sample.unsqueeze(0))
output_ensemble = torch.mean(output, dim=0)
pred_masks = F.softmax(output_ensemble, dim=1)
pred_masks = torch.argmax(pred_masks, dim=1)
show_image(pred_masks.squeeze())
show_image(mask)
