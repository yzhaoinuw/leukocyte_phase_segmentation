# -*- coding: utf-8 -*-
"""
Created on Mon May  8 11:52:29 2023

@author: Yue
"""

import torch
from torch import Tensor
import matplotlib.pyplot as plt

def show_image(img: torch.Tensor, cmap="gray"):
    plt.imshow(img, cmap="gray")
    plt.show()
    
def calculate_dice(confmat: Tensor, num_class=4, smooth=1):
    dice = {i: (2 * confmat[i, i] + smooth) / (confmat[i, :].sum() + confmat[:, i].sum() + smooth) for i in range(num_class)}
    return dice
        