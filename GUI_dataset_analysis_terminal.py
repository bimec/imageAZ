"""
Created on Fri Jul  5 11:39:29 2024

title: GUI dataset analysis software
description: GUI software for implementing, analysing and validating the solution Semi-supervised approach for segmenting tumours in MRI Images.

@author: wickramw
@github: https://github.com/bimec/image-AZ/tree/deployment
"""

print("Loading..!")

from PyQt5.QtGui import QPixmap, QGuiApplication
from PyQt5.QtCore import  Qt
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QFileDialog, QGraphicsScene, QInputDialog, QWidget, QGraphicsPixmapItem
from PyQt5 import uic, QtCore, QtWidgets

import sys

import GUI.main as GUI
print("GUI Imported")

# Open the software
app = QApplication(sys.argv)
window = GUI.UI()
window.setWindowTitle("Image-AZ GUI")
window.show() 
app.exec()
print("App Closed!")





