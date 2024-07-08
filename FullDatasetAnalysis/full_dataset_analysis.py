# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 11:39:29 2024

title: 
description:

@author: wickramw
@github: https://github.com/dilshan-n-wickramarachchi
"""
import nibabel as nib
import cv2
import os
import numpy as np
import csv

# Algorithms
print("Loading modules and data..!")
from ROIProcessing import ROIProcessing
from TumourSegmenting import segmentTumour

# =============================================================================
# %% Load dataset
# =============================================================================
# MRI
f_mri  = r"C:\Users\wickramw\OneDrive - London South Bank University\Imaging-AZ\dataset-types\dataset-orig\train\45_post_24h_rare.nii.gz"
# f_mri = r"{}".format(input("Enter the path for image: "))
mri_data = nib.load(f_mri)
mri_data = mri_data.get_fdata()


# Masks
f_mask = r"C:\Users\wickramw\OneDrive - London South Bank University\Imaging-AZ\dataset-types\dataset-orig\train\21_post_15min_rare.nii.gz"
# f_mask = r"{}".format(input("Enter the path for label: "))
# f_mask = f_mask = f_mri.split("train")[0] + "train_labels" +f_mri.split("train")[1]
mask_data = nib.load(f_mask)
mask_data = mask_data.get_fdata()

# =============================================================================
# %% Save data as images 
# =============================================================================
print("Savin raw images..!")
save_folder = r'D:\image-AZ\myanalysis'
try:
    save_folder_output = save_folder + '/{}_output'.format(f_mri.split('\\')[-1])
    os.mkdir(save_folder_output)
except:
    pass

# save MRI images
try:
    save_folder_output_rawMri = save_folder_output + '/{}_rawMRI'.format(f_mri.split('\\')[-1])
    os.mkdir(save_folder_output_rawMri)
    
    for i in range(mri_data.shape[-1]):
        img_name = save_folder_output_rawMri + '/rawImg_Z{}.png'.format(i)
        img_save = mri_data[:,:,i]/mri_data[:,:,i].max()*255
        img_save = img_save.astype(np.uint8)
        cv2.imwrite(img_name, img_save)
except Exception as e:
    print(e)

# save Mask images
try:
    save_folder_output_rawMask = save_folder_output + '/{}_rawMask'.format(f_mask.split('\\')[-1])
    os.mkdir(save_folder_output_rawMask)
    
    for i in range(mask_data.shape[-1]):
        img_name = save_folder_output_rawMask + '/maskImg_Z{}.png'.format(i)
        img_save = mask_data[:,:,i]/mask_data[:,:,i].max()*255
        img_save = img_save.astype(np.uint8)
        cv2.imwrite(img_name, img_save)
except Exception as e:
    print(e)

# =============================================================================
# %% Calculate ROI
# =============================================================================
print("Calculating ROIs..!")
roiProcessor = ROIProcessing()
roi_data = np.zeros_like(mri_data)
for i in range(mri_data.shape[-1]):
    img = roiProcessor.prepImage(mri_data[:,:,i])    # Preprocess
    cluster_img = roiProcessor.applyClustering(img)  # Clustering 
    img_c = roiProcessor.enhanceROI(cluster_img)     # Enhance ROI
    roi_mask = roiProcessor.getROIMask(img_c)        # Get ROI Mask
    img_ROI = roiProcessor.applyMask(roi_mask)       # Get ROI Image   
    roi_data[:,:,i] = img_ROI

# =============================================================================
# %% Save ROI
# =============================================================================
print("Saving ROIs..!")
# save ROI images
try:
    save_folder_output_roiMri = save_folder_output + '/{}_roiMRI'.format(f_mri.split('\\')[-1])
    os.mkdir(save_folder_output_roiMri)
    
    for i in range(roi_data.shape[-1]):
        img_name = save_folder_output_roiMri + '/roiImg_Z{}.png'.format(i)
        img_save = roi_data[:,:,i]/roi_data[:,:,i].max()*255
        img_save = img_save.astype(np.uint8)
        cv2.imwrite(img_name, img_save)
except Exception as e:
    print(e)

# =============================================================================
# %% Predict Tumour 
# =============================================================================
print("Looking for tumours..")
# load segmentTumour
model_path = r"MRI_seg_best.pt"
model = segmentTumour(model_path)

x_input = []
for i in range(roi_data.shape[-1]):
    image = cv2.cvtColor(roi_data[:,:,i].astype(np.uint8), cv2.COLOR_GRAY2RGB)
    x_input.append(image)
results = model.predTumour(x_input)

# Segmentation
header_summary = ['file', 'name', 'class', 'confidence', 'box', 'segments']
summary = []
for i in range(len(x_input)):
    print(i)
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
       
# DICE Score
header_dice = ['file', 'dice']
dice = []
for i in range(len(x_input)):
    d = {header_dice[0] : 'roiImg_Z{}.png'.format(i)}   
    img_mask = results[i].cpu().masks
    if(img_mask == None):
        d[header_dice[1]] = 0/np.sum(mask_data[:,:,i])
    else:
        img_mask = img_mask.data.numpy()
        img_mask = (np.sum(img_mask, axis=0) > 0)
        d[header_dice[1]] = 2*np.sum(img_mask * mask_data[:,:,i])/(np.sum(img_mask) + np.sum(mask_data[:,:,i]))
    dice.append(d)

# Tumour plots
header_predImg = ['file', 'predImg']
predImg = []
for i in range(len(x_input)):
    img = {header_predImg[0] : 'roiImg_Z{}.png'.format(i)} 
    img[header_predImg[1]] = results[i].plot()
    predImg.append(img)

# Tumour features
header_tumFeatures = ['file', 'area', 'perimeter', 'center']
tfeatures = []
for i in range(len(x_input)):
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
        



# =============================================================================
# %% Save results
# =============================================================================
## results summary
# summary csv   : imgName, box, class, confidence, class name, segments
# dice_score
# image: pred with tumour

print("Saving Predictions..!")

try:
    # CSVs
    save_folder_output_resultsMri = save_folder_output + '/{}_resultsMRI'.format(f_mri.split('\\')[-1])
    os.mkdir(save_folder_output_resultsMri)
    
    # summary csv
    keys = summary[0].keys()
    with open(save_folder_output_resultsMri + '/summary.csv', 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, keys)
        dict_writer.writeheader()
        dict_writer.writerows(summary)
    
    # dice_score
    keys = dice[0].keys()
    with open(save_folder_output_resultsMri + '/dice_score.csv', 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, keys)
        dict_writer.writeheader()
        dict_writer.writerows(dice)
    
    # # Segmentation images
    save_folder_output_resultspredImg = save_folder_output + '/{}_predImg'.format(f_mri.split('\\')[-1])
    os.mkdir(save_folder_output_resultspredImg)
    
    # predImages
    for i in range(len(predImg)):
        img_name = save_folder_output_resultspredImg + '/predImg_Z{}.png'.format(i)
        cv2.imwrite(img_name, predImg[i]['predImg'])
    
except Exception as e:
    print(e)