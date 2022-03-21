#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 22:41:45 2022

@author: alex
"""
import nibabel as nb
import matplotlib.pyplot as plt

title = ['1st', '2nd','3rd','4th','5th','6th','7th','8th','9th']
def display9Images(top):
# Show subplots | shape: (3,3) 
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15,15))
    for i, ax in enumerate(axs.flatten()):
        plt.sca(ax)
        img = nb.load(top[i][1])
        img = img.get_fdata()
        plt.imshow(img[:, :, img.shape[2] // 2].T, cmap='gray')
        plt.axis('off')
        #plt.colorbar()
        plt.title('{} related image with distance: {} \nLabel: {} handed'.format(title[i],format(top[i][0],".4f"),top[i][2]))

    #plt.tight_layout()
    plt.suptitle('Top Nine Related search for the query image')
    plt.show()