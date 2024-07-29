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
    
    def predTumour(self, image, imgsz=320, conf=0.25):        
        self.results = self.model.predict(image, imgsz=320, conf=0.5)
        return self.results
    
    
    def segmentSummary(self):
        # Segmentation
        results = self.results
        header_summary = ['file', 'name', 'class', 'confidence', 'box', 'segments']
        summary = []
        for i in range(len(results)):
            # print(i)
            img_res = results[i].cpu().summary()
            row = dict.fromkeys(header_summary)
            
            if(len(img_res) > 0):
                for t in range(len(img_res)):
                    row[header_summary[0]] = 'roiImg_Z{}.png'.format(i)   
                    for ele in header_summary[1:]:
                        row[ele] = img_res[t][ele]
                    summary.append(row)
            else:
                row[header_summary[0]] = 'roiImg_Z{}.png'.format(i)   
                for ele in header_summary[1:]:
                    row[ele] = '-'
                summary.append(row)
        
        return summary
    
    def diceScore(self, mask):
        # DICE Score
        results = self.results
        header_dice = ['file', 'dice']
        dice = []
        for i in range(len(results)):
            d = {header_dice[0] : 'roiImg_Z{}.png'.format(i)}   
            img_mask = results[i].cpu().masks
            if(img_mask == None):
                d[header_dice[1]] = 0 if(np.sum(mask[:,:,i]) !=0) else np.nan
            else:
                img_mask = img_mask.data.numpy()
                img_mask = (np.sum(img_mask, axis=0) > 0)
                d[header_dice[1]] = 2*np.sum(img_mask * mask[:,:,i])/(np.sum(img_mask) + np.sum(mask[:,:,i]))
            dice.append(d)
        return dice
    
    def tumourPlots(self):
        # Tumour plots
        results = self.results
        header_predImg = ['file', 'predImg']
        predImg = []
        for i in range(len(results)):
            img = {header_predImg[0] : 'roiImg_Z{}.png'.format(i)} 
            img[header_predImg[1]] = results[i].plot()
            predImg.append(img)
            
        return predImg


    def tumourFeatures(self):
        # Tumour features
        results = self.results
        header_tumFeatures = ['file', 'area', 'perimeter', 'center']
        tfeatures = []
        for i in range(len(results)):
            img_mask = results[i].cpu().masks
            row = {}
            if(img_mask != None):
                row[header_tumFeatures[0]] = 'roiImg_Z{}.png'.format(i)
                
                img_mask = img_mask.data.numpy()
                img_mask = (np.sum(img_mask, axis=0) > 0)
                
                contours, hierarchy = cv2.findContours(img_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                
                area = []
                perimeter = []
                centers = []
                for i in range(len(contours)):
                    cnt = contours[i]
                    area.append(cv2.contourArea(cnt))
                    perimeter.append(cv2.arcLength(cnt,True))
                    if(area[-1]>0):
                        M = cv2.moments(cnt)
                        centers.append((int(M['m10']/M['m00']), int(M['m01']/M['m00'])))
                    else:
                        centers.append('-')
                
                row[header_tumFeatures[1]] = area
                row[header_tumFeatures[2]] = perimeter
                row[header_tumFeatures[3]] = centers  
                tfeatures.append(row)
            
            else:
               row[header_tumFeatures[0]] = 'roiImg_Z{}.png'.format(i)   
               for ele in header_tumFeatures[1:]:
                   row[ele] = '-'
               tfeatures.append(row)
        
        return tfeatures
        

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
    mask_data = cv2.imread(label_path, cv2.COLOR_BGR2GRAY)/255
    mask_data = mask_data.reshape(mask_data.shape[0], mask_data.shape[1], 1)
    
    # load segmentTumour
    model = segmentTumour(model_path)
    
    # results
    results = model.predTumour(image)
    summary = model.segmentSummary()
    dice = model.diceScore(mask_data)
    predImg = model.tumourPlots()
    tfeatures = model.tumourFeatures()
    
    ## plots
    plt.figure(3)
    ax = plt.subplot(1,3,1)
    plt.title('Ground Tumour Mask')
    plt.imshow(mask_data[:,:,0])
    
    ax = plt.subplot(1,3,2)
    plt.title('Raw Image')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    
    ax = plt.subplot(1,3,3)
    plt.title('Predicted Tumour Mask')
    plt.imshow(predImg[0]['predImg'])
    
    

