# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 16:04:17 2024

title: 
description:

@author: wickramw
@github: https://github.com/dilshan-n-wickramarachchi
"""

# =============================================================================
# Import libraries
# =============================================================================
import nibabel as nib
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

from ultralytics import YOLO

class segmentTumour:
    
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.image = None
        self.label = None
        self.mask = None
        return
    
    def predTumour2(self, image, imgsz=320, conf=0.25):        
        results = self.model.predict(image, imgsz=320, conf=0.5)
        for i in range(len(results)):
            full_mask = np.zeros_like(image[:, :, 0])
            masks = results[i].masks.data.cpu().numpy().astype(np.uint8)
            n_masks = len(masks)
            for i in range(n_masks):
                full_mask = full_mask | masks[i]
            self.full_mask = full_mask
        return self.full_mask, n_masks, results
    
    def predTumour(self, image, imgsz=320, conf=0.25):        
        results = self.model.predict(image, imgsz=320, conf=0.5)
        
        return results
    
    def valdiateTumour(self, label):        
        self.label = label
        dice_score = 2*(np.sum(self.full_mask*self.label))/(np.sum(self.full_mask) + np.sum(self.label))   
        return dice_score
    
    def drawMask(self, image, mask_generated, color=[0,0,150]):
        # masked_image = image.copy()
        # image += 10
        try:
            masked_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
        except Exception as e:
            # print(e)
            masked_image = image.copy()
            
        mask_generated = cv2.cvtColor(self.full_mask, cv2.COLOR_GRAY2RGB)
        
        print(mask_generated.shape, masked_image.shape)
        
        masked_image = np.where(mask_generated.astype(int), np.array(color, dtype='uint8'), masked_image)
        masked_image = masked_image.astype(np.uint8)
        return cv2.addWeighted(image, 1, masked_image, 0.5, 0)

# =============================================================================
#%% Tumour detection per file
# =============================================================================
if __name__ == '__main__':
    # Load data
    model_path = r"C:/Users/wickramw/OneDrive - London South Bank University/Imaging-AZ/image-AZ-git/image-AZ/model/MRI_seg_best.pt"
    image_path = r'C:/Users/wickramw/OneDrive - London South Bank University/Imaging-AZ/Software/New folder/roi/ROI_Image_z-7.png.png'
    label_path = r'C:/Users/wickramw/OneDrive - London South Bank University/Imaging-AZ/Software/New folder/mask/mask_z-7.png'
    
    # load images
    image = cv2.imread(image_path)
    label = cv2.imread(label_path, cv2.COLOR_BGR2GRAY)/255
    
    # load segmentTumour
    model = segmentTumour(model_path)
    
    # results
    masks, n_masks, results = model.predTumour(image)
    dice_score = model.valdiateTumour(label)
    print('Dice Score: {}'.format(dice_score))
    
    # get mask
    mask_image = model.drawMask(image, masks)
    
    ## plots
    plt.figure(3)
    ax = plt.subplot(2,2,1)
    plt.title('Ground Tumour Mask')
    plt.imshow(label)
    
    ax = plt.subplot(2,2,2)
    plt.title('Predicted Tumour Mask')
    plt.imshow(masks)
    
    ax = plt.subplot(2,2,3)
    plt.title('Raw Image')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    
    ax = plt.subplot(2,2,4)
    plt.title('Tumour Image')
    plt.imshow(mask_image)#cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)*masks)

    # cv2.imshow('img', mask_image)

