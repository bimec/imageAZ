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

from sklearn.cluster import KMeans

# =============================================================================
#%% Load dataset
# =============================================================================
# Image
f_image  = r"dataset\train\21_post_15min_rare.nii.gz"
# f_image = r"{}".format(input("Enter the path for image: "))
img_data = nib.load(f_image)
img_data = img_data.get_fdata()


# label
f_label = r"dataset\train_labels\21_post_15min_rare.nii.gz"
# f_label = r"{}".format(input("Enter the path for label: "))
label_data = nib.load(f_label)
label_data = label_data.get_fdata()

# =============================================================================
#%% Plot Raw Data 
# =============================================================================
# select layer
img_x = 0; #img_x = input("Enter X layer: ")
img_y = 0; #img_y = input("Enter Y layer: ")
img_z = 7; #img_z = input("Enter Z layer: ")

# Plots
fig = plt.figure(1)
plt.suptitle("Raw data analysis")

ax = plt.subplot(2,6,1)
plt.title('Raw Image - Layer X={}'.format(img_x))
plt.imshow(img_data[img_x, :, : ])
plt.xticks([])
plt.yticks([])

ax = plt.subplot(2,6,2)
plt.title('Raw Image - Layer Y={}'.format(img_y))
plt.imshow(img_data[:, img_y, :].T)
plt.xticks([])
plt.yticks([])

ax = plt.subplot(2,6,3)
plt.title('Raw Image - Layer Z={}'.format(img_z))
plt.imshow(img_data[:, :, img_z])
plt.xticks([])
plt.yticks([])

ax = plt.subplot(2,6,4)
plt.title('Tumour Image - Layer X={}'.format(img_x))
plt.imshow((img_data*label_data)[img_x, :, : ])
plt.xticks([])
plt.yticks([])

ax = plt.subplot(2,6,5)
plt.title('Tumour Image - Layer Y={}'.format(img_y))
plt.imshow((img_data*label_data)[:, img_y, :].T)
plt.xticks([])
plt.yticks([])

ax = plt.subplot(2,6,6)
plt.title('Tumour Image - Layer Z={}'.format(img_z))
plt.imshow((img_data*label_data)[:, :, img_z])
plt.xticks([])
plt.yticks([])

ax = plt.subplot(2,6,7)
plt.title('Raw Histogram - Layer X={}'.format(img_x))
plt.hist(img_data[img_x, :, : ].flatten(), bins=255, density=None, weights=None)
plt.xlabel("Pixel intensity")
plt.ylabel("Count")

ax = plt.subplot(2,6,8)
plt.title('Raw Histogram - Layer Y={}'.format(img_y))
plt.hist(img_data[ :, img_y, : ].flatten(), bins=255, density=None, weights=None)
plt.xlabel("Pixel intensity")
plt.ylabel("Count")

ax = plt.subplot(2,6,9)
plt.title('Raw Histogram - Layer Z={}'.format(img_z))
plt.hist(img_data[:, :, img_z].flatten(), bins=255, density=None, weights=None)
plt.xlabel("Pixel intensity")
plt.ylabel("Count")

ax = plt.subplot(2,6,10)
plt.title('Tumour Histogram - Layer X={}'.format(img_x))
plt.hist((img_data*label_data)[img_x, :, :].flatten(), bins=255, density=None, weights=None)
plt.xlabel("Pixel intensity")
plt.ylabel("Count")

ax = plt.subplot(2,6,11)
plt.title('Tumour Histogram - Layer Y={}'.format(img_y))
plt.hist((img_data*label_data)[:, img_y, :].flatten(), bins=255, density=None, weights=None)
plt.xlabel("Pixel intensity")
plt.ylabel("Count")

ax = plt.subplot(2,6,12)
plt.title('Tumour Histogram - Layer Z={}'.format(img_z))
plt.hist((img_data*label_data)[:, :, img_z].flatten(), bins=255, density=None, weights=None)
plt.xlabel("Pixel intensity")
plt.ylabel("Count")


# =============================================================================
#%% Adaptive noise removal
# =============================================================================
#img_z = input("Enter Z layer for processing: ")
img = img_data[:, :, img_z]

# Preprocess
img_d = (img - img.min())/(img.max() - img.min())*255
img_d = cv2.resize(img_d, (320*4, 320*4))
img_d = cv2.bilateralFilter(img_d.astype(np.uint8), 10, 10, 2)
x = img_d.flatten().reshape(-1,1)

## Clustering 
n_clusters = 6
kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
y_pred = kmeans.fit_predict(x)

# Update cluter labels
c = np.round(kmeans.cluster_centers_).astype(int)
clusters = {c[i][0]:i for i in range(n_clusters)}
y_pred_ordered = np.zeros_like(y_pred)
c = np.sort(c[:, 0])
for i, v in enumerate(c):
    y_pred_ordered[y_pred==clusters[v]] = i
c = y_pred_ordered.reshape(img_d.shape)

## Enhance ROI 
img_c = (c == 0).astype(np.uint8)
img_c = cv2.erode(img_c, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=2).astype(np.uint8)
img_c = cv2.medianBlur(img_c, 31)
img_c = cv2.dilate(img_c, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)), iterations=1) 
img_c = 1-img_c

## Separate ROI
img_roi_mask = np.zeros_like(img_c)
contours, hierarchy = cv2.findContours(img_c, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
contours_ext = []
area_all = [cv2.contourArea(contours[c]) for c in range(len(hierarchy[0]))]
areas_d = []
for c in range(len(hierarchy[0])):
    if(hierarchy[0, c, -1] == -1):
        if(area_all[c] > np.max(area_all)*0.01):
            areas_d.append(area_all[c])
            contours_ext.append(contours[c])
            cv2.drawContours(img_roi_mask, contours, c, (255,255,255), thickness=cv2.FILLED)
img_roi_mask = img_roi_mask.astype(int)/255

# Map to raw image
img_roi_mask = cv2.resize(img_roi_mask, img.shape)
img_roi = img*img_roi_mask
img_roi = (img_roi)/(img_roi.max())*255

# =============================================================================
#%% Plot Adaptive Noise removal 
# =============================================================================
fig = plt.figure(2)
plt.suptitle("Adaptive Noise removal")

ax = plt.subplot(3,2,1)
plt.title('Raw Image - Layer Z={}'.format(img_z))
plt.imshow(img_data[:, :, img_z])
plt.xticks([])
plt.yticks([])

ax = plt.subplot(3,2,2)
plt.title('Raw Histogram - Layer Z={}'.format(img_z))
plt.hist(img_data[:, :, img_z].flatten(), bins=255, density=None, weights=None)
plt.xlabel("Pixel intensity")
plt.ylabel("Count")

ax = plt.subplot(3,2,3)
plt.title('ROI Mask - Layer Z={}'.format(img_z))
plt.imshow(img_roi_mask)
plt.xlabel('Yellow - ROI, Purple - Background')
plt.xticks([])
plt.yticks([])

ax = plt.subplot(3,2,4)
plt.title('Background Mask - Layer Z={}'.format(img_z))
plt.imshow(img_roi_mask + label_data[:, :, img_z])
plt.xlabel('Yellow - Tumour, Green - ROI, Purple - Background')
plt.xticks([])
plt.yticks([])

ax = plt.subplot(3,2,5)
plt.title('ROI Image and Tumour - Layer Z={}'.format(img_z))
plt.imshow(img_roi)

ax = plt.subplot(3,2,6)
plt.title('ROI Histogram - Layer Z={}'.format(img_z))
plt.hist(img_roi.flatten(), bins=255, range=(1,img_roi.max()), density=None, weights=None)
plt.xlabel("Pixel intensity")
plt.ylabel("Count")


# =============================================================================
#%% Tumour detection per file
# =============================================================================
from ultralytics import YOLO
model_path = r"C:/Users/wickramw/OneDrive - London South Bank University/Imaging-AZ/Models/MRI_seg_best.pt"
image = cv2.cvtColor(img_roi.astype(np.uint8), cv2.COLOR_GRAY2RGB)

model = YOLO(model_path)
results = model.predict(image, imgsz=320, conf=0.25)
masks = results[0].masks.data.cpu().numpy()[0]

# results
label = label_data[:,:,img_z]
dice_score = 2*(np.sum(masks*label))/(np.sum(masks) + np.sum(label))
print('DICE Score: {}'.format(dice_score))

## plots
plt.figure(3)
ax = plt.subplot(2,2,1)
plt.title('Ground Tumour Mask - Layer Z={}'.format(img_z))
plt.imshow(label)

ax = plt.subplot(2,2,2)
plt.title('Predicted Tumour Mask - Layer Z={}'.format(img_z))
plt.imshow(masks)

ax = plt.subplot(2,2,3)
plt.title('Raw Image - Layer Z={}'.format(img_z))
plt.imshow(img)

ax = plt.subplot(2,2,4)
plt.title('Tumour Image - Layer Z={}'.format(img_z))
plt.imshow(img*masks)




