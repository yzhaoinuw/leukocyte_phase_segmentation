# -*- coding: utf-8 -*-
"""
Created on Mon May  8 11:52:29 2023

@author: Yue
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt


def show_image(img: torch.Tensor, cmap="gray"):
    plt.imshow(img, cmap=cmap)
    plt.show()


def calculate_dice(confmat: torch.Tensor, num_class=4, smooth=1):
    dice = {
        i: (2 * confmat[i, i] + smooth)
        / (confmat[i, :].sum() + confmat[:, i].sum() + smooth)
        for i in range(num_class)
    }
    return dice


def calculate_accuracy(confmat: torch.Tensor):
    accuracy = torch.sum(torch.diag(confmat)) / torch.sum(confmat)
    return accuracy.item()


def dice_loss(output: torch.Tensor, masks: torch.Tensor, smooth=1):
    pred_masks = F.softmax(output, dim=1)
    one_hot_masks = (F.one_hot(masks, num_classes=4)).permute((0, 3, 1, 2))
    intersection = pred_masks * one_hot_masks
    area_total = pred_masks + one_hot_masks
    intersection_sum = torch.sum(intersection, dim=(-1, -2))
    area_total_sum = torch.sum(area_total, dim=(-1, -2))
    dice = (2 * intersection_sum + smooth) / (area_total_sum + smooth)
    return 1 - torch.mean(dice)


class SobelTransform:
    def __init__(self):
        self.sobel_kernel_x = (
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.sobel_kernel_y = (
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
        )

    def __call__(self, image):
        # Ensure the image is a PyTorch tensor. If not, convert it.
        if not isinstance(image, torch.Tensor):
            image = transforms.ToTensor()(image)

        # Apply the Sobel filter
        G_x = F.conv2d(image.unsqueeze(0), self.sobel_kernel_x, padding=1)
        G_y = F.conv2d(image.unsqueeze(0), self.sobel_kernel_y, padding=1)

        # Compute the gradient magnitude and remove the batch dimension
        G_magnitude = torch.sqrt(G_x**2 + G_y**2).squeeze(0)

        return G_magnitude
