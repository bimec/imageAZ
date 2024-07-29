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

class ROIProcessing:
    
    def __init__(self, n_clusters = 6):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
        self.kmeans_trained = False
        return 
    
    def trainClusters(self, x):
        x = x.flatten().reshape(-1,1)
        y_pred = self.kmeans.fit_predict(x)
        self.kmeans_trained = True
        return y_pred
    
    def prepImage(self, image, res = (4, 4), filt1 = (10, 10, 2)):
        self.raw_img = image
        if(self.raw_img.max()>0):
            img = (self.raw_img - self.raw_img.min())/(self.raw_img.max() - self.raw_img.min())*255
        img = cv2.resize(img, (320*res[0], 320*res[1]))
        img = cv2.bilateralFilter(img.astype(np.uint8), *filt1) #filt1[0], filt1[1], filt1[2]
        return img
    
    def applyClustering(self, img):
        x = img.flatten().reshape(-1,1)
        
        # Clustering
        if(self.kmeans_trained):
            y_pred = self.kmeans.predict(x)
            self.kmeans_trained = True
        else:
            y_pred = self.kmeans.fit_predict(x)
        y_pred = y_pred.reshape(img.shape)
        
        # Update cluter labels
        c = np.round(self.kmeans.cluster_centers_).astype(int)
        clusters = {c[i][0]:i for i in range(self.n_clusters)}
        y_pred_ordered = np.zeros_like(y_pred)
        c = np.sort(c[:, 0])
        for i, v in enumerate(c):
            y_pred_ordered[y_pred==clusters[v]] = i
        cluster_img = y_pred_ordered.reshape(img.shape)
        
        return cluster_img
    
    def enhanceROI(self, cluster_img, filt2 = (3,3,2), filt3 = (31), filt4=(7,7,1)):
        img_c = (cluster_img == 0).astype(np.uint8)
        img_c = cv2.erode(img_c, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(filt2[0],filt2[1])), iterations=filt2[2]).astype(np.uint8)
        img_c = cv2.medianBlur(img_c, filt3)
        img_c = cv2.dilate(img_c, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(filt4[0],filt4[1])), iterations=filt4[2]) 
        img_c = 1-img_c
        return img_c
    
    def getROIMask(self, img_c, filt5=(0.01)):
        img_roi_mask = np.zeros_like(img_c)
        contours, hierarchy = cv2.findContours(img_c, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours_ext = []
        area_all = [cv2.contourArea(contours[c]) for c in range(len(hierarchy[0]))]
        areas_d = []
        for c in range(len(hierarchy[0])):
            if(hierarchy[0, c, -1] == -1):
                if(area_all[c] > np.max(area_all)*filt5):
                    areas_d.append(area_all[c])
                    contours_ext.append(contours[c])
                    cv2.drawContours(img_roi_mask, contours, c, (255,255,255), thickness=cv2.FILLED)
        
        if(img_roi_mask.max()>0):
            img_roi_mask = img_roi_mask/img_roi_mask.max()*255
        img_roi_mask = img_roi_mask.astype(np.uint8)
        self.img_roi_mask = cv2.resize(img_roi_mask, self.raw_img.shape)
        
        return self.img_roi_mask
    
    def applyMask(self, roi_mask):
        img_ROI = self.img_roi_mask.astype(float) * self.raw_img
        
        if(img_ROI.max()>0):
            img_ROI = (img_ROI)/(img_ROI.max())*255
        self.img_ROI = img_ROI.astype(np.uint8)
        return self.img_ROI
    
    def getROI(self, image):
        img = self.prepImage(image)
        cluster_img = self.applyClustering(img)
        img_c = self.enhanceROI(cluster_img)
        roi_mask = self.getROIMask(img_c)
        img_ROI = self.applyMask(roi_mask)
        return img_ROI

# =============================================================================
#%% Code Example 
# =============================================================================

if __name__ == '__main__':
    # =============================================================================
    # Load dataset
    # =============================================================================
    # Image
    f_image  = r"C:\Users\wickramw\OneDrive - London South Bank University\Imaging-AZ\dataset-types\dataset-orig\train\21_post_15min_rare.nii.gz"
    # f_image = r"{}".format(input("Enter the path for image: "))
    img_data = nib.load(f_image)
    img_data = img_data.get_fdata()
    
    
    # label
    f_label = r"C:\Users\wickramw\OneDrive - London South Bank University\Imaging-AZ\dataset-types\dataset-orig\train_labels\21_post_15min_rare.nii.gz"
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
    image = img_data[:, :, img_z]
    
    ## ROI Processing
    roiProcessor = ROIProcessing()
    
    
    img = roiProcessor.prepImage(image)              # Preprocess
    cluster_img = roiProcessor.applyClustering(img)  # Clustering 
    img_c = roiProcessor.enhanceROI(cluster_img)     # Enhance ROI
    roi_mask = roiProcessor.getROIMask(img_c)        # Get ROI Mask
    img_ROI = roiProcessor.applyMask(roi_mask)       # Get ROI Image   
    
    
    
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
    plt.imshow(roi_mask)
    plt.xlabel('Yellow - ROI, Purple - Background')
    plt.xticks([])
    plt.yticks([])
    
    ax = plt.subplot(3,2,4)
    plt.title('ROI Image and Tumour- Layer Z={}'.format(img_z))
    plt.imshow(roi_mask/255 + label_data[:, :, img_z])
    plt.xlabel('Yellow - Tumour, Green - ROI, Purple - Background')
    plt.xticks([])
    plt.yticks([])
    
    ax = plt.subplot(3,2,5)
    plt.title('ROI Image - Layer Z={}'.format(img_z))
    plt.imshow(img_ROI)
    
    ax = plt.subplot(3,2,6)
    plt.title('ROI Histogram - Layer Z={}'.format(img_z))
    plt.hist(img_ROI.flatten(), bins=255, range=(1,img_ROI.max()), density=None, weights=None)
    plt.xlabel("Pixel intensity")
    plt.ylabel("Count")





