# Image-AZ

## Quick Start Guidelines
Code: https://github.com/bimec/image-AZ 
The Python codes are made available via the above GitHub link. The following libraries are used for implementing the algorithms and please follow the instructions in the link for installation. The PyQt5 library is used to develop the GUI software and it is only required if GUI software is used for analysis. 

* Numpy	For numerical operations	<a href="https://numpy.org/install/" target="_blank">https://numpy.org/install/</a> 
* Matplotlib 	For generating plots	<a href="https://matplotlib.org/stable/install/index.html" target="_blank">https://matplotlib.org/stable/install/index.html/</a> 
* Sci-kit learn	For K-means clustering	<a href="https://scikit-learn.org/stable/install.html" target="_blank">https://scikit-learn.org/stable/install.html</a> 
* Opencv	For handling images	<a href="https://opencv.org/get-started/" target="_blank">https://opencv.org/get-started/</a> 
* Nibabel	For reading .nii images	<a href="https://nipy.org/nibabel/installation.html" target="_blank">https://nipy.org/nibabel/installation.html</a> 
* Pytorch-GPU	Backend for YOLO network	<a href="https://pytorch.org/get-started/locally/" target="_blank">https://pytorch.org/get-started/locally/</a> 
* Ultralytics	Implementing YOLO network	<a href="https://docs.ultralytics.com/quickstart/" target="_blank">https://docs.ultralytics.com/quickstart/</a> 

## GUI Software 
The GUI software allows the analysis of each data file with the plots. There are 3 tabs,
1. **The Raw Data Tab**:- Allows to view the MRI data as 2D images and export them
2. **The Adaptive Filters Tab**:- Here, the user needs to import raw 2D MRI images exported in the 1st tab. The software can extract and export the region-of-interest (ROI) images.
3. **Tumour Detection Tab**:- Here, the user needs to import 2D ROI images exported in the 2nd tab. Then the software can segment the tumour regions and calculate the features of the tumours.

## Full Data Analysis Code
This script executes the three main stages of the tumour extraction algorithm, 1.) reading the data, 2.) extracting the ROI and 3.) segmenting the tumour regions. 
Before executing the script, please check whether the input file location and output file locations are properly assigned.

## Disclaimer 
Not indented for medical diagnosis

