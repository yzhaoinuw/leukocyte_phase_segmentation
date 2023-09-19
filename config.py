# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 17:25:39 2023

@author: Yue
"""

mask_mapping = {
    209: 0,
    188: 3,
    157: 2,
    76: 1,
}  # 0: background, 3: probing, 2: phase dark, 1: phase bright
mask_mapping_reversed = {0: 209, 3: 188, 2: 157, 1: 76}
