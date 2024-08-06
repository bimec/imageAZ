# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 21:54:30 2024

title: 
description:

@author: wickramw
@github: https://github.com/dilshan-n-wickramarachchi
"""

from PyQt5.QtGui import QPixmap, QGuiApplication
from PyQt5.QtCore import  Qt
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QFileDialog, QGraphicsScene, QInputDialog, QWidget, QGraphicsPixmapItem
from PyQt5 import uic, QtCore, QtWidgets

import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
plt.ioff()

import nibabel as nib
import cv2
import os
from pathlib import Path
import numpy as np
import csv

# Algorithms
os.chdir('.')
from Algorithms.ROIProcessing import ROIProcessing
from Algorithms.TumourSegmenting import segmentTumour
model_path = Path("Algorithms/MRI_seg_best.pt")
model = segmentTumour(model_path)

class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100, projection='2d'):
        plt.close('all')
        if(width!=0):
            self.fig = plt.figure(figsize=(width, height), dpi=dpi)
        else:
            self.fig = plt.figure(dpi=dpi)
        plt.close('all')
        
        if(projection == '3d'):
            self.axes = self.fig.add_subplot(111, projection=projection)
        else:
            self.axes = self.fig.add_subplot(111)
            
        super(MplCanvas, self).__init__(self.fig)  
        
        return


class UI(QMainWindow, QApplication):
    ##UI initialization
    def __init__(self):
        ## Set ui
        super(UI, self).__init__()
        self.ui = uic.loadUi(Path("GUI/DNW_Imaging-main.ui"), self)
        self.ui.tabWidget.setCurrentIndex(0)
        
        ## Buttons
        # Tab 1
        self.ui.btn_loadMRIData.clicked.connect(self.loadMRI)
        self.ui.btn_loadMRIMask.clicked.connect(self.loadMask)
        self.ui.btn_exportMRIImages.clicked.connect(self.exportMRI)
        self.ui.btn_clearMRI.clicked.connect(self.clearRaw)
        
        # Tab 2
        self.ui.btn_loadImage.clicked.connect(self.loadImage)
        self.ui.btn_getROI.clicked.connect(self.getROI)
        self.ui.btn_mapTumour.clicked.connect(self.mapTumour)
        self.ui.btn_exportROI.clicked.connect(self.exportROI)
        self.ui.btn_clearROI.clicked.connect(self.clearROIPlots)
        
        # Tab 3
        self.ui.btn_loadROI.clicked.connect(self.loadROI)
        self.ui.btn_detectTumour.clicked.connect(self.detectTumour)
        self.ui.btn_validate.clicked.connect(self.validate)
        self.ui.btn_tFeature.clicked.connect(self.getTFeatures)
        self.ui.btn_clearTumour.clicked.connect(self.clearTumour)
        self.ui.btn_exportTumour.clicked.connect(self.saveSummary)
        
        ## SpinBox
        self.ui.sb_rawX.valueChanged.connect(self.updateRSB_X)
        self.ui.sb_rawY.valueChanged.connect(self.updateRSB_Y)
        self.ui.sb_rawZ.valueChanged.connect(self.updateRSB_Z)
        self.ui.sb_tumourX.valueChanged.connect(self.updateTSB_X)
        self.ui.sb_tumourY.valueChanged.connect(self.updateTSB_Y)
        self.ui.sb_tumourZ.valueChanged.connect(self.updateTSB_Z)
        
        
        ## Algorithms
        self.roiProcessor = ROIProcessing()
        self.model = segmentTumour(model_path)
        return
    
    def updateRSB_X(self):
        plt.close('all')
        self.ui.sb_rawX.setDisabled(True)
        img_data_yz = self.data_d[self.ui.sb_rawX.value(), :, :]
        self.setScene(self.ui.plot_ImgYZ, img_data_yz, 'img', dlayout= False)   
        self.setScene(self.ui.plot_HistYZ, img_data_yz, 'hist', dlayout= True) # YZ   
        self.ui.sb_rawZ.setDisabled(False)
        return
    
    def updateRSB_Z(self):
        plt.close('all')
        self.ui.sb_rawZ.setDisabled(True)
        img_data_xy = self.data_d[:, :, self.ui.sb_rawZ.value()] #Z slider 
        self.setScene(self.ui.plot_ImgXY, img_data_xy, 'img', dlayout= True) # XY     
        self.setScene(self.ui.plot_HistXY, img_data_xy, 'hist', dlayout= True) # XY
        self.ui.sb_rawZ.setDisabled(False)
        return
    
    def updateRSB_Y(self):
        plt.close('all')
        self.ui.sb_rawY.setDisabled(True)
        img_data_zx = self.data_d[:, self.ui.sb_rawY.value(), :]
        self.setScene(self.ui.plot_ImgZX, img_data_zx.T, 'img', dlayout= False)       
        self.setScene(self.ui.plot_HistZX, img_data_zx, 'hist', dlayout= True) #ZX        
        elf.ui.sb_rawY.setDisabled(False)
        return
    
    def updateTSB_X(self):
        plt.close('all')
        self.ui.sb_tumourX.setDisabled(True)
        img_data_tyz = (self.data_dl*self.data_d)[self.ui.sb_tumourX.value(), :, :] # X slider
        self.setScene(self.ui.plot_TImgYZ, img_data_tyz, 'img', dlayout= False) # YZ
        self.setScene(self.ui.plot_THistYZ, img_data_tyz, 'hist', dlayout= True, range=(1, max(2, img_data_tyz.max()))) # YZ      
        elf.ui.sb_tumourX.setDisabled(False)
        return
    
    def updateTSB_Y(self):
        plt.close('all')
        self.ui.sb_tumourY.setDisabled(True)
        img_data_tzx = (self.data_dl*self.data_d)[:, self.ui.sb_tumourY.value(), :] #Y Slider
        self.setScene(self.ui.plot_TImgZX, img_data_tzx.T, 'img', dlayout= False) #ZX
        self.setScene(self.ui.plot_THistZX, img_data_tzx, 'hist', dlayout= True, range=(1, max(2, img_data_tzx.max()))) #ZX      
        elf.ui.sb_tumourY.setDisabled(False)
        return
    
    def updateTSB_Z(self):  
        plt.close('all')  
        self.ui.sb_tumourZ.setDisabled(True)
        img_data_txy = (self.data_dl*self.data_d)[:, :, self.ui.sb_tumourZ.value()] #Z slider 
        self.setScene(self.ui.plot_TImgXY, img_data_txy, 'img', dlayout= True) # XY
        self.setScene(self.ui.plot_THistXY, img_data_txy, 'hist', dlayout= True, range=(1, max(2, img_data_txy.max()))) # XY      
        elf.ui.sb_tumourZ.setDisabled(False)
        return
    
    def setScene(self, scene, data, dtype, **kwargs):
        # Canvas
        canvas = MplCanvas(self, dpi=50, projection='2d')
        
        if(dtype == 'img'):
            canvas.axes.imshow(np.flip(data, axis=0), cmap='gist_earth', interpolation='nearest', origin='lower')
            canvas.axes.get_xaxis().set_ticks([]);   
            canvas.axes.get_yaxis().set_ticks([])
            canvas.fig.canvas.draw()
            
        elif(dtype == 'hist'):
            if('range' in kwargs.keys()):
                canvas.axes.hist(data.flatten(), bins=255, range=kwargs['range'], density=None, weights=None)
            else:
                canvas.axes.hist(data.flatten(), bins=255, density=None, weights=None)
            
        # Plot Styling
        if('dlayout' in kwargs.keys()):
            if(kwargs['dlayout']):
                canvas.fig.tight_layout()

        
        # Create layout
        layout = self.clearScene(scene)
        layout.addWidget(canvas)
        scene.setLayout(layout) 

        try:
            plt.close(canvas.fig)
            del(canvas)
        except:
            pass

        
        return scene
    
    def clearScene(self, scene):
        layout = scene.layout()
        if(layout != None):
            for i in reversed(range(layout.count())): 
                layout.itemAt(i).widget().setParent(None)
        else:
            layout = QtWidgets.QVBoxLayout()
        return layout
        
    
    def loadMRI(self):
        self.clearRaw()
        file = QFileDialog.getOpenFileName(self, 'Single File', '', '*.nii.gz')
        
        if(file[1]!=''):
            file = Path(file[0])
            if(file.exists()):
                self.ui.label_MRIFile.setText(file.name)
                
                img_dose = nib.load(file)
                self.data_d = img_dose.get_fdata()
                
                ## slider 
                img_data_xy = self.data_d[:, :, 0] #Z slider 
                img_data_yz = self.data_d[0, :, :] # X slider
                img_data_zx = self.data_d[:, 0, :] #Y Slider
                
                ## Plots - Raw
                self.setScene(self.ui.plot_ImgXY, img_data_xy, 'img', dlayout= True) # XY
                self.setScene(self.ui.plot_ImgYZ, img_data_yz, 'img', dlayout= False) # YZ
                self.setScene(self.ui.plot_ImgZX, img_data_zx.T, 'img', dlayout= False) #ZX
        
                ## Histograms - Raw
                self.setScene(self.ui.plot_HistXY, img_data_xy, 'hist', dlayout= True) # XY
                self.setScene(self.ui.plot_HistYZ, img_data_yz, 'hist', dlayout= True) # YZ
                self.setScene(self.ui.plot_HistZX, img_data_zx, 'hist', dlayout= True) #ZX
       
        return
    
    def loadMask(self):    
        file = QFileDialog.getOpenFileName(self, 'Single File', '', '*.nii.gz')

        if(file[1]!=''):
            file = Path(file[0])
            if(file.exists()):
                self.ui.label_MRIMask.setText(file.name)
                
                img_dl = nib.load(file)
                self.data_dl = img_dl.get_fdata()
                
                ## slider 
                img_data_txy = (self.data_dl*self.data_d)[:, :, 0] #Z slider 
                img_data_tyz = (self.data_dl*self.data_d)[0, :, :] # X slider
                img_data_tzx = (self.data_dl*self.data_d)[:, 0, :] #Y Slider
                
                ## Plots - Tumour
                self.setScene(self.ui.plot_TImgXY, img_data_txy, 'img', dlayout= True) # XY
                self.setScene(self.ui.plot_TImgYZ, img_data_tyz, 'img', dlayout= False) # YZ
                self.setScene(self.ui.plot_TImgZX, img_data_tzx.T, 'img', dlayout= False) #ZX
        
                ## Histograms - Raw
                self.setScene(self.ui.plot_THistXY, img_data_txy, 'hist', dlayout= True, range=(1, max(2, img_data_txy.max()))) # XY
                self.setScene(self.ui.plot_THistYZ, img_data_tyz, 'hist', dlayout= True, range=(1, max(2, img_data_tyz.max()))) # YZ
                self.setScene(self.ui.plot_THistZX, img_data_tzx, 'hist', dlayout= True, range=(1, max(2, img_data_tzx.max()))) #ZX
        return
    
    def exportMRI(self):
        file = QFileDialog.getExistingDirectory(self, "Select Directory")

        if(file !=''):
            file = Path(file)

            if(file.exists()):
                try:
                    raw_file = Path(file, 'raw')
                    raw_file.mkdir()
                    mask_file = Path(file, 'mask')
                    mask_file.mkdir()
                    tumour_file = Path(file, 'tumour_label')
                    tumour_file.mkdir()
                    
                    save_folder = raw_file
                    n_images = self.data_d.shape[-1]
                    for img_i in range(n_images):
                        img_data_xy = self.data_d[:,:,img_i]
                        if(img_data_xy.max()>0):
                            img_data_xy = img_data_xy/img_data_xy.max()*255 if(img_data_xy.max()!=0) else img_data_xy
                        img_name = Path(save_folder, 'raw_z-{}.png'.format(img_i))
                        cv2.imwrite(img_name, img_data_xy)
                        
                    save_folder = mask_file
                    n_images = (self.data_dl).shape[-1]
                    for img_i in range(n_images):
                        img_data_xy = (self.data_dl)[:,:,img_i]
                        if(img_data_xy.max()>0):
                            img_data_xy = img_data_xy/img_data_xy.max()*255 if(img_data_xy.max()!=0) else img_data_xy
                        img_name = Path(save_folder, 'mask_z-{}.png'.format(img_i))
                        cv2.imwrite(img_name, img_data_xy)
                    
                    save_folder = tumour_file
                    n_images = (self.data_dl*self.data_d).shape[-1]
                    for img_i in range(n_images):
                        img_data_xy = (self.data_dl*self.data_d)[:,:,img_i]
                        if(img_data_xy.max()>0):
                            img_data_xy = img_data_xy/img_data_xy.max()*255 if(img_data_xy.max()!=0) else img_data_xy
                        img_name = Path(save_folder, 'tumour_z-{}.png'.format(img_i))
                        cv2.imwrite(img_name, img_data_xy)
                         
                except Exception as e:
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Critical)
                    msg.setWindowTitle("Export Error")
                    msg.setText(str(e))
                    msg.exec_()
                    msg.show()
                
                else:
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Information)
                    msg.setWindowTitle("Sucessful")
                    msg.setText('Export Completed! Export to {}'.format(file))
                    msg.exec_()
                    msg.show()
            
        return
    
    def clearRaw(self):
        # Spinbox
        self.ui.sb_rawX.setValue(0)
        self.ui.sb_rawY.setValue(0)
        self.ui.sb_rawZ.setValue(0)
        self.ui.sb_tumourX.setValue(0)
        self.ui.sb_tumourY.setValue(0)
        self.ui.sb_tumourZ.setValue(0)
        
        # labels
        self.ui.label_MRIFile.setText('-')
        self.ui.label_MRIMask.setText('-')
        
        ## Plots - Raw
        self.clearScene(self.ui.plot_ImgXY) # XY
        self.clearScene(self.ui.plot_ImgYZ) # YZ
        self.clearScene(self.ui.plot_ImgZX) # ZX
        
        # ## Histograms - Raw
        self.clearScene(self.ui.plot_HistXY) # XY
        self.clearScene(self.ui.plot_HistYZ) # YZ
        self.clearScene(self.ui.plot_HistZX) # ZX
        
        ## Plots - Tumour
        self.clearScene(self.ui.plot_TImgXY) # XY
        self.clearScene(self.ui.plot_TImgYZ) # YZ
        self.clearScene(self.ui.plot_TImgZX) # ZX

        # ## Histograms - Tumour
        self.clearScene(self.ui.plot_THistXY) # XY
        self.clearScene(self.ui.plot_THistYZ) # YZ
        self.clearScene(self.ui.plot_THistZX) # ZX
        return
    
    def loadImage(self):
        self.clearROIPlots()
        file = QFileDialog.getOpenFileName(self, 'Single File', '', "Images (*.png *.xpm *.jpg)")
        
        if(file[1] !=''):
            file = Path(file[0])    
    
            if(file.exists()):
                self.image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
                self.ui.label_imageName.setText(file.name)
                
                self.setScene(self.ui.plot_origImg, self.image, 'img', dlayout= True) 
                self.setScene(self.ui.plot_origHist, self.image, 'hist', dlayout= True) 
        return
    
    def getROI(self):
        img = self.roiProcessor.prepImage(self.image)              # Preprocess
        cluster_img = self.roiProcessor.applyClustering(img)  # Clustering 
        img_c = self.roiProcessor.enhanceROI(cluster_img)     # Enhance ROI 
        self.roi_mask = self.roiProcessor.getROIMask(img_c)        # Get ROI Mask
        self.img_ROI = self.roiProcessor.applyMask(self.roi_mask)       # Get ROI Image   
        
        self.setScene(self.ui.plot_mask, self.roi_mask, 'img', dlayout= True) 
        self.setScene(self.ui.plot_ROIImage, self.img_ROI, 'img', dlayout= True) 
        self.setScene(self.ui.plot_ROIHist, self.img_ROI, 'hist', dlayout= True, range=(1, self.img_ROI.max())) 
        
        return
    
    def mapTumour(self):
        file = QFileDialog.getOpenFileName(self, 'Single File', '', "Images (*.png *.xpm *.jpg)")
        
        if(file[1] !=''):
            file = Path(file[0])
            if(file.exists()):
                self.image_tumour = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
                
                if(self.roi_mask.max()>0):
                    img_regions = self.roi_mask/self.roi_mask.max() 
                if(self.image_tumour.max()>0):
                    img_regions += self.image_tumour/self.image_tumour.max()
                if(img_regions.max()>0):
                    img_regions = img_regions/img_regions.max()*255
                
                self.setScene(self.ui.plot_tumour, img_regions, 'img', dlayout= True) 
        return
    
    def exportROI(self):
        file = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        if(file !=''):
            file = Path(file)
            if(file.exists()):
                try:
                    # os.mkdir(file + r"/roi")
                    save_folder = file 
                    img_save = self.img_ROI/self.img_ROI.max()*255 if(self.img_ROI.max() != 0) else self.img_ROI
                    img_name = Path(save_folder, 'ROI_Image_{}.png'.format(self.ui.label_imageName.text().split('_')[-1]))
                    #print(img_name)
                    cv2.imwrite(img_name, img_save)
                except Exception as e:
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Critical)
                    msg.setWindowTitle("Export Error!")
                    msg.setText(str(e))
                    msg.exec_()
                    msg.show()
                else:
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Information)
                    msg.setWindowTitle("Sucessful")
                    msg.setText('Export Completed! Export to {}'.format(file))
                    msg.exec_()
                    msg.show()
        return
    
    def clearROIPlots(self):
        # labels
        self.ui.label_imageName.setText('-')
        
        # Plots
        self.clearScene(self.ui.plot_origImg)
        self.clearScene(self.ui.plot_origHist)
        self.clearScene(self.ui.plot_mask)
        self.clearScene(self.ui.plot_tumour)
        self.clearScene(self.ui.plot_ROIImage)
        self.clearScene(self.ui.plot_ROIHist)
        return
    
    def loadROI(self):
        file = QFileDialog.getOpenFileName(self, 'Single File', '', "Images (*.png *.xpm *.jpg)")
        if(file[1] !=''):
            file = Path(file[0])
            if(file.exists()):
                self.imageROI = cv2.imread(file)
                self.ui.label_ROIName.setText(file.name)
                self.setScene(self.ui.plot_TumourROIImage, cv2.cvtColor(self.imageROI, cv2.COLOR_BGR2GRAY), 'img', dlayout= True) 
        return
    
    def detectTumour(self):
        # get mask
        self.results = self.model.predTumour(self.imageROI)
        self.summary = self.model.segmentSummary()
        self.predImg = self.model.tumourPlots()
        
        
        # mask
        img_mask = self.results[0].cpu().masks
        if(img_mask != None):
            img_mask = img_mask.data.numpy()
            
            self.ui.label_ntumour.setText(str(len(img_mask)))
            
            img_mask = (np.sum(img_mask, axis=0) > 0)
            img_tumor = cv2.cvtColor(self.imageROI, cv2.COLOR_BGR2GRAY) * img_mask
        else:
            img_mask = 0
            img_tumor = cv2.cvtColor(self.imageROI, cv2.COLOR_BGR2GRAY) * img_mask
            self.ui.label_ntumour.setText("No Tumour")
        
        # plot
        self.setScene(self.ui.plot_predImage, img_tumor, 'img', dlayout= True)         
        
        # hist
        self.setScene(self.ui.plot_THist, img_tumor, 'hist', dlayout= True, range=(1, max(img_tumor.max(),5))) 
        return
    
    def validate(self):
        file = QFileDialog.getOpenFileName(self, 'Single File', '', "Images (*.png *.xpm *.jpg)")
        
        if(file[1] !=''):
            file = Path(file[0])
            if(file.exists()):
                label = cv2.imread(file, cv2.COLOR_BGR2GRAY)/255
                label = label.reshape(label.shape[0], label.shape[1], 1)
                
                if(self.results[0].cpu().masks != None):
                    
                    self.dice_score = self.model.diceScore(label)
                    self.ui.label_dice.setText(str(np.format_float_positional(self.dice_score[0]['dice'], precision=3)))
                    # print('Dice Score: {}'.format(dice_score))
                else:
                    self.ui.label_dice.setText('nan')
                    
                # plot
                self.setScene(self.ui.plot_valImage, self.predImg[0]['predImg'], 'img', dlayout= True)  
        return
    
    def getTFeatures(self):
        if(self.results[0].cpu().masks != None):
            self.tfeatures = self.model.tumourFeatures()
        
            area = np.round(self.tfeatures[0]['area'])
            perimeter = np.round(self.tfeatures[0]['perimeter'])
            centers = np.round(self.tfeatures[0]['center'])
            
            # update labels
            self.ui.label_tArea.setText(str(area))
            self.ui.label_tPerimeter.setText(str(perimeter))
            self.ui.label_tCenters.setText(str(centers))
            
        else:
            # update labels
            self.ui.label_tArea.setText('nan')
            self.ui.label_tPerimeter.setText('nan')
            self.ui.label_tCenters.setText('nan')
            
        return
    
    def clearTumour(self):
        # labels
        self.ui.label_ROIName.setText('-')
        self.ui.label_ntumour.setText('-')
        self.ui.label_dice.setText('-')
        self.ui.label_tArea.setText('-')
        self.ui.label_tPerimeter.setText('-')
        self.ui.label_tCenters.setText('-')
        
        # Plots
        self.clearScene(self.ui.plot_TumourROIImage)
        self.clearScene(self.ui.plot_predImage)
        self.clearScene(self.ui.plot_THist)
        self.clearScene(self.ui.plot_valImage)
        return

    def saveSummary(self):
        # CSVs
        file = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        
        if(file !=''):
            file = Path(file)
        
            if(file.exists()):
                save_folder_output_resultsMri = file
                # dice_score
                if( hasattr(self, 'dice_score') ):
                    keys = self.dice_score[0].keys()
                    with open(Path(save_folder_output_resultsMri, 'dice_score.csv'), 'w', newline='') as f:
                        dict_writer = csv.DictWriter(f, keys)
                        dict_writer.writeheader()
                        dict_writer.writerows(self.dice_score)
            
                # Tumour features
                if( hasattr(self, 'tfeatures') ):
                    keys = self.tfeatures[0].keys()
                    with open(Path(save_folder_output_resultsMri, 'tfeatures.csv'), 'w', newline='') as f:
                        dict_writer = csv.DictWriter(f, keys)
                        dict_writer.writeheader()
                        dict_writer.writerows(self.tfeatures)
                
                # Segmentation images
                for i in range(len(self.predImg)):
                    img_name = Path(save_folder_output_resultsMri, 'predImg_Z{}.png'.format(i))
                    cv2.imwrite(img_name, self.predImg[0]['predImg'])
                    
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setWindowTitle("Sucessful")
                msg.setText('Export Completed! Export to {}'.format(file))
                msg.exec_()
                msg.show()
        return
    
    def closeEvent(self, a):
        QApplication.quit()
        return
    

#-------------------Create Application-------------------------------
def main():
    app = QApplication(sys.argv)
    window = UI()
    window.setWindowTitle("Image-AZ GUI")
    window.show() 
    app.exec()
    

#-------------------Main Method--------------------------------------
if __name__ == '__main__':
    main()
    