# Image-AZ

## Quick Start Guidelines
Code: https://github.com/bimec/image-AZ 
The Python codes are made available via the above GitHub link. The following libraries are used for implementing the algorithms and please follow the instructions in the link for installation. The PyQt5 library is used to develop the GUI software and it is only required if GUI software is used for analysis. 

* Numpy	For numerical operations	https://numpy.org/install/ 
* Matplotlib 	For generating plots	https://matplotlib.org/stable/install/index.html
* Sci-kit learn	For K-means clustering	https://scikit-learn.org/stable/install.html
* Opencv	For handling images	https://opencv.org/get-started/
* Nibabel	For reading .nii images	https://nipy.org/nibabel/installation.html
* Pytorch-GPU	Backend for YOLO network	https://pytorch.org/get-started/locally/
* Ultralytics	Implementing YOLO network	https://docs.ultralytics.com/quickstart/

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

