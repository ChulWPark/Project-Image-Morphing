#! /user/local/bin/python3.4

# Import PySide classes
import sys
from PySide.QtCore import *
from PySide.QtGui import *

from MorphingGUI import *

import os
import numpy as np
import imageio
from scipy import ndimage
from PIL import Image, ImageDraw
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from matplotlib.cm import *
import os.path

class MorphConsumer(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MorphConsumer, self).__init__(parent)
        self.setupUi(self)
        
        # Member variables
        self.startImage = None
        self.startImagePath = None
        self.startPoints = None
        self.endImage = None
        self.endImagePath = None
        self.endPoints = None
        self.nop = None
        self.delaunay = None
        self.startLabel = None
        self.endLabel = None
        self.ready1 = False
        self.ready2 = False
        self.mode = None
        self.tempxs = None
        self.tempys = None
        self.tempxe = None
        self.tempye = None
        self.newpick = False
        self.stupid = False
        self.blockS = False
        self.blockE = True
        self.left = 0
        self.deleteone = False
        self.deletetwo = False
        self.special = False
        self.reload = False

        # Initial State
        self.lineEdit.setText(str(float(0))) 
        self.lineEdit.setAlignment(Qt.AlignCenter)
        self.lineEdit.setEnabled(False)
        self.btnBlend.setEnabled(False)
        self.checkBox.setEnabled(False)
        self.horizontalSlider.setEnabled(False)

        # Button Clicks
        self.btnLoadS.clicked.connect(self.loadDataS)
        self.btnLoadE.clicked.connect(self.loadDataE)
        self.btnBlend.clicked.connect(self.blendImages)
        self.horizontalSlider.valueChanged.connect(self.displayAlpha)
        self.checkBox.stateChanged.connect(self.delaunayEnable)
        self.centralwidget.mousePressEvent = self.outside
        self.graphicsView_3.mousePressEvent = self.outside

    def outside(self, event):
        if self.ready1 == True:
            self.ready2 = True
            self.drawDots(0)
            self.drawDots(1)
            self.ready1 = False
            self.ready2 = False
            self.newpick = True
            if self.checkBox.isChecked():
                self.delaunayEnable()
        
    def drawDots(self, flag):
        if flag == 0:
            self.startPoints = np.loadtxt(self.startImagePath + '.txt')
            fig = plt.figure()
            ax = plt.axes([0, 0, 1, 1])
            ax.xaxis.set_ticks_position('top')
            ax.invert_yaxis()
            if len(self.startImage.shape) == 2:
                ax.imshow(self.startImage, extent=[0, self.startImage.shape[1], self.startImage.shape[0], 0], cmap=gray)
            else:
                ax.imshow(self.startImage, extent=[0, self.startImage.shape[1], self.startImage.shape[0], 0])
            if len(self.startPoints.shape) == 1:
                if self.ready2 == True:
                    self.left = 0
                    if self.startPoints != []:
                        ax.plot(self.startPoints[0], self.startPoints[1], 'o', markeredgecolor='blue')
                    line = "{:6d}{:6d}\n".format(self.tempxs, self.tempys)
                    with open(self.startImagePath + '.txt', 'a') as myFile:
                        myFile.write(line)
                    self.startPoints = np.loadtxt(self.startImagePath + '.txt')
                    ax.plot(self.tempxs, self.tempys, 'o', color='blue', markeredgecolor='blue')
                else:
                    self.left = 1
                    if self.deleteone == True:
                        self.left = 0
                        self.deleteone = False
                        if len(self.startPoints) == 2:
                            ax.plot(self.startPoints[0], self.startPoints[1], 'o', color='blue', markeredgecolor='blue')
                        pass
                    elif len(self.startPoints) == 0:
                        self.left = 1
                        ax.plot(self.tempxs, self.tempys, 'o', color='lawngreen', markeredgecolor='lawngreen')
                    elif len(self.startPoints) == 2:
                        self.left = 1
                        ax.plot(self.startPoints[0], self.startPoints[1], 'o', color='blue', markeredgecolor='blue')
                        ax.plot(self.tempxs, self.tempys, 'o', color='lawngreen', markeredgecolor='lawngreen')
            else:
                if self.ready2 == True:
                    self.left = 0
                    line = "{:6d}{:6d}\n".format(self.tempxs, self.tempys)
                    with open(self.startImagePath + '.txt', 'a') as myFile:
                        myFile.write(line)
                    self.startPoints = np.loadtxt(self.startImagePath + '.txt')
                    ax.plot(self.startPoints[:,0], self.startPoints[:,1], 'o', color='blue', markeredgecolor='blue')
                    if self.mode == 2:
                        ax.plot(self.startPoints[:self.nop,0], self.startPoints[:self.nop,1], 'o', color='red', markeredgecolor='red')
                        ax.plot(self.startPoints[self.nop:,0], self.startPoints[self.nop:,1], 'o', color='blue', markeredgecolor='blue')
                elif self.stupid == True:
                    self.left = 0
                    self.stupid = False
                    self.startPoints = np.loadtxt(self.startImagePath + '.txt')
                    ax.plot(self.startPoints[:,0], self.startPoints[:,1], 'o', color='blue', markeredgecolor='blue')
                    if self.mode == 2:
                        ax.plot(self.startPoints[:self.nop,0], self.startPoints[:self.nop,1], 'o', color='red', markeredgecolor='red')
                        ax.plot(self.startPoints[self.nop:,0], self.startPoints[self.nop:,1], 'o', color='blue', markeredgecolor='blue')
                else:
                    self.left = 1
                    self.startPoints = np.loadtxt(self.startImagePath + '.txt')
                    ax.plot(self.startPoints[:,0], self.startPoints[:,1], 'o', color='blue', markeredgecolor='blue')
                    ax.plot(self.tempxs, self.tempys, 'o', color='lawngreen', markeredgecolor='lawngreen')
                    if self.mode == 2:
                        ax.plot(self.startPoints[:self.nop,0], self.startPoints[:self.nop,1], 'o', color='red', markeredgecolor='red')
                        ax.plot(self.startPoints[self.nop:,0], self.startPoints[self.nop:,1], 'o', color='blue', markeredgecolor='blue')
                        ax.plot(self.tempxs, self.tempys, 'o', color='lawngreen', markeredgecolor='lawngreen')
            ax.set_axis_off()
            plt.savefig('temp.png')
            plt.close()
            self.startLabel = QLabel(self.graphicsView_1)
            pm = QPixmap('temp.png')
            pm = pm.scaled(400,300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.startLabel.setPixmap(pm)
            self.startLabel.mousePressEvent = self.mcEventS
            self.graphicsView_1.keyPressEvent = self.kpEventS
            self.startLabel.show()
            os.remove('temp.png')
        elif flag == 1:
            self.endPoints = np.loadtxt(self.endImagePath + '.txt')
            fig = plt.figure()
            ax = plt.axes([0, 0, 1, 1])
            ax.xaxis.set_ticks_position('top')
            ax.invert_yaxis()
            if len(self.endImage.shape) == 2:
                ax.imshow(self.endImage, extent=[0, self.endImage.shape[1], self.endImage.shape[0], 0], cmap=gray)
            else:
                ax.imshow(self.endImage, extent=[0, self.endImage.shape[1], self.endImage.shape[0], 0])
            if len(self.endPoints.shape) == 1:
                if self.ready2 == True or len(self.startPoints.shape) != 1:
                    self.left = 0
                    self.ready2 = False
                    if self.endPoints != []:
                        ax.plot(self.endPoints[0], self.endPoints[1], 'o', color='blue', markeredgecolor='blue')
                    line = "{:6d}{:6d}\n".format(self.tempxe, self.tempye)
                    with open(self.endImagePath + '.txt', 'a') as myFile:
                        myFile.write(line)
                    self.endPoints = np.loadtxt(self.endImagePath + '.txt')
                    ax.plot(self.tempxe, self.tempye, 'o', color='blue', markeredgecolor='blue')
                else:
                    if self.deletetwo == True:
                        self.left = 1
                        self.deletetwo = False
                        if len(self.endPoints) == 2:
                            ax.plot(self.endPoints[0], self.endPoints[1], 'o', color='blue', markeredgecolor='blue')
                        pass
                    elif len(self.endPoints) == 0:
                        self.left = 2
                        ax.plot(self.tempxe, self.tempye, 'o', color='lawngreen', markeredgecolor='lawngreen')
                    elif len(self.endPoints) == 2:
                        self.left = 2
                        ax.plot(self.tempxe, self.tempye, 'o', color='lawngreen', markeredgecolor='lawngreen')
                        ax.plot(self.endPoints[0], self.endPoints[1], 'o', color='blue', markeredgecolor='blue')
            else:
                if self.ready2 == True:
                    self.left = 0
                    self.ready2 = False
                    line = "{:6d}{:6d}\n".format(self.tempxe, self.tempye)
                    with open(self.endImagePath + '.txt', 'a') as myFile:
                        myFile.write(line)
                    self.endPoints = np.loadtxt(self.endImagePath + '.txt')
                    ax.plot(self.endPoints[:,0], self.endPoints[:,1], 'o', color='blue', markeredgecolor='blue')
                    if self.mode == 2:
                        ax.plot(self.endPoints[:self.nop,0], self.endPoints[:self.nop,1], 'o', color='red', markeredgecolor='red')
                        ax.plot(self.endPoints[self.nop:,0], self.endPoints[self.nop:,1], 'o', color='blue', markeredgecolor='blue')
                elif self.newpick == True:
                    self.newpick = False
                    self.endPoints = np.loadtxt(self.endImagePath + '.txt')
                    ax.plot(self.endPoints[:,0], self.endPoints[:,1], 'o', color='blue', markeredgecolor='blue')
                    if self.mode == 2:
                        ax.plot(self.endPoints[:self.nop,0], self.endPoints[:self.nop,1], 'o', color='red', markeredgecolor='red')
                        ax.plot(self.endPoints[self.nop:,0], self.endPoints[self.nop:,1], 'o', color='blue', markeredgecolor='blue')
                else:
                    self.left = 2
                    self.endPoints = np.loadtxt(self.endImagePath + '.txt')
                    ax.plot(self.endPoints[:,0], self.endPoints[:,1], 'o', color='blue', markeredgecolor='blue')
                    ax.plot(self.tempxe, self.tempye, 'o', color='lawngreen', markeredgecolor='lawngreen')
                    if self.mode == 2:
                        ax.plot(self.endPoints[:self.nop,0], self.endPoints[:self.nop,1], 'o', color='red', markeredgecolor='red')
                        ax.plot(self.endPoints[self.nop:,0], self.endPoints[self.nop:,1], 'o', color='blue', markeredgecolor='blue')
                        ax.plot(self.tempxe, self.tempye, 'o', color='lawngreen', markeredgecolor='lawngreen')
            ax.set_axis_off()
            plt.savefig('temp.png')
            plt.close()
            self.endLabel = QLabel(self.graphicsView_2)
            pm = QPixmap('temp.png')
            pm = pm.scaled(400,300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.endLabel.setPixmap(pm)
            self.endLabel.mousePressEvent = self.mcEventE
            self.graphicsView_2.keyPressEvent = self.kpEventE
            self.endLabel.show()
            os.remove('temp.png')

    def kpEventS(self, event):
        if event.key() == Qt.Key_Backspace:
            if self.left == 1 or self.left == 2:
                if self.startPoints.ndim == 1:
                    self.deleteone = True
                    self.newpick == False
                else:
                    self.newpick = True
                self.stupid = True
                self.ready2 = False
                self.drawDots(0)
                self.blockS = False
                self.blockE = True
                self.stupid = False
                if self.checkBox.isChecked():
                    self.delaunayEnable()
            else:
                pass
    
    def kpEventE(self, event):
        if event.key() == Qt.Key_Backspace:
            if self.left == 1 or self.left == 2:
                if self.endPoints.ndim == 1:
                    self.deletetwo = True
                    self.newpick = False
                else:
                    self.newpick = True
                self.ready2 = False
                self.drawDots(1)
                self.blockE = False
                self.blockS = True
                if self.checkBox.isChecked():
                    self.special = True
                    self.delaunayEnable()
            else:
                pass

    def mcEventS(self, event):
        if self.blockS == False:
            if self.mode == 0:
                self.mode = 2
            if self.ready1 == True:
                line = "{:6d}{:6d}\n".format(self.tempxs, self.tempys)
                with open(self.startImagePath + '.txt', 'a') as myFile:
                    myFile.write(line)
                line = "{:6d}{:6d}\n".format(self.tempxe, self.tempye)
                with open(self.endImagePath + '.txt', 'a') as myFile:
                    myFile.write(line)
                self.newpick = True
            self.ready1 = False
            xpos = round(event.x() * self.startImage.shape[1] / 400)
            ypos = round(event.y() * self.startImage.shape[0] / 300)
            self.tempxs = xpos
            self.tempys = ypos
            self.drawDots(0)
            if isinstance(self.endPoints, np.ndarray):
                self.drawDots(1)
            if self.checkBox.isChecked():
                self.delaunayEnable()
            self.blockS = True
            self.blockE = False

    def mcEventE(self, event):
        if self.blockE == False:
            self.ready1 = True
            xpos = round(event.x() * self.endImage.shape[1] / 400)
            ypos = round(event.y() * self.endImage.shape[0] / 300)
            self.tempxe = xpos
            self.tempye = ypos
            self.drawDots(1)
            if self.checkBox.isChecked():
                self.delaunayEnable()
            self.blockE = True
            self.blockS = False

    def delaunayEnable(self):
        if self.checkBox.isChecked():
            self.delaunay = Delaunay(self.startPoints)
            # Display Delaunay Triangles On startImage
            fig = plt.figure()
            ax = plt.axes([0, 0, 1, 1])
            ax.xaxis.set_ticks_position('top')
            ax.invert_yaxis()
            if len(self.startImage.shape) == 2:
                ax.imshow(self.startImage, extent=[0, self.startImage.shape[1], self.startImage.shape[0], 0], cmap=gray)
            else:
                ax.imshow(self.startImage, extent=[0, self.startImage.shape[1], self.startImage.shape[0], 0])
            if self.mode == 0:
                with np.errstate(invalid='ignore'):
                    ax.triplot(self.startPoints[:,0], self.startPoints[:,1], self.delaunay.simplices.copy(), color='red')
                ax.plot(self.startPoints[:,0], self.startPoints[:,1], 'o', color='red', markeredgecolor='red')
                if self.reload == True:
                    with np.errstate(invalid='ignore'):
                        ax.triplot(self.startPoints[:,0], self.startPoints[:,1], self.delaunay.simplices.copy(), color='cyan')
                    ax.plot(self.startPoints[:self.nop,0], self.startPoints[:self.nop,1], 'o', color='red', markeredgecolor='red')
                    ax.plot(self.startPoints[self.nop:,0], self.startPoints[self.nop:,1], 'o', color='blue', markeredgecolor='blue')
            elif self.mode == 1:
                with np.errstate(invalid='ignore'):
                    ax.triplot(self.startPoints[:,0], self.startPoints[:,1], self.delaunay.simplices.copy(), color='blue')
                ax.plot(self.startPoints[:,0], self.startPoints[:,1], 'o', color='blue', markeredgecolor='blue')
                if self.left == 1 or self.left == 2:
                    ax.plot(self.tempxs, self.tempys, 'o', color='lawngreen', markeredgecolor='lawngreen')
            elif self.mode == 2:
                with np.errstate(invalid='ignore'):
                    ax.triplot(self.startPoints[:,0], self.startPoints[:,1], self.delaunay.simplices.copy(), color='cyan')
                ax.plot(self.startPoints[:self.nop,0], self.startPoints[:self.nop,1], 'o', color='red', markeredgecolor='red')
                ax.plot(self.startPoints[self.nop:,0], self.startPoints[self.nop:,1], 'o', color='blue', markeredgecolor='blue')
                if self.left == 1 or self.left == 2:
                    ax.plot(self.tempxs, self.tempys, 'o', color='lawngreen', markeredgecolor='lawngreen')
            ax.set_axis_off()
            plt.savefig('temp1.png')
            plt.close()
            self.startLabel = QLabel(self.graphicsView_1)
            pm = QPixmap('temp1.png')
            pm = pm.scaled(400,300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.startLabel.setPixmap(pm)
            self.startLabel.show()
            self.startLabel.mousePressEvent = self.mcEventS
            self.startLabel.keyPressEvent = self.kpEventS
            os.remove('temp1.png')
            # Display Delaunay Triangles On endImage
            fig = plt.figure()
            ax = plt.axes([0, 0, 1, 1])
            ax.xaxis.set_ticks_position('top')
            ax.invert_yaxis()
            if len(self.startImage.shape) == 2:
                ax.imshow(self.endImage, extent=[0, self.endImage.shape[1], self.endImage.shape[0], 0], cmap=gray)
            else:
                ax.imshow(self.endImage, extent=[0, self.endImage.shape[1], self.endImage.shape[0], 0])
            if self.mode == 0:
                with np.errstate(invalid='ignore'):
                    ax.triplot(self.endPoints[:,0], self.endPoints[:,1], self.delaunay.simplices.copy(), color='red')
                ax.plot(self.endPoints[:,0], self.endPoints[:,1], 'o', color='red', markeredgecolor='red')
                if self.reload == True:
                    ax.triplot(self.endPoints[:,0], self.endPoints[:,1], self.delaunay.simplices.copy(), color='cyan')
                    ax.plot(self.endPoints[:self.nop,0], self.endPoints[:self.nop,1], 'o', color='red', markeredgecolor='red')
                    ax.plot(self.endPoints[self.nop:,0], self.endPoints[self.nop:,1], 'o', color='blue', markeredgecolor='blue')
            elif self.mode == 1:
                with np.errstate(invalid='ignore'):
                    ax.triplot(self.endPoints[:,0], self.endPoints[:,1], self.delaunay.simplices.copy(), color='blue')
                ax.plot(self.endPoints[:,0], self.endPoints[:,1], 'o', color='blue', markeredgecolor='blue')
                if self.special == True:
                    self.special = False
                elif self.left == 2:
                    ax.plot(self.tempxe, self.tempye, 'o', color='lawngreen', markeredgecolor='lawngreen')
            elif self.mode == 2:
                with np.errstate(invalid='ignore'):
                    ax.triplot(self.endPoints[:,0], self.endPoints[:,1], self.delaunay.simplices.copy(), color='cyan')
                ax.plot(self.endPoints[:self.nop,0], self.endPoints[:self.nop,1], 'o', color='red', markeredgecolor='red')
                ax.plot(self.endPoints[self.nop:,0], self.endPoints[self.nop:,1], 'o', color='blue', markeredgecolor='blue')
                if self.left == 2:
                    ax.plot(self.tempxe, self.tempye, 'o', color='lawngreen', markeredgecolor='lawngreen')
            ax.set_axis_off()
            plt.savefig('temp2.png')
            plt.close()
            self.endLabel = QLabel(self.graphicsView_2)
            pm = QPixmap('temp2.png')
            pm = pm.scaled(400,300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.endLabel.setPixmap(pm)
            self.endLabel.show()
            self.endLabel.mousePressEvent = self.mcEventE
            self.endLabel.keyPressEvent = self.kpEventE
            os.remove('temp2.png')
        else:
            if self.mode == 0:
                self.loadDataFromFileS(self.startImagePath)
                self.loadDataFromFileE(self.endImagePath)
            elif self.mode == 1 or self.mode == 2: 
                if self.left == 0:
                    self.stupid = True
                    self.ready2 = False
                    self.newpick = True
                elif self.left == 1:
                    self.ready2 = False
                    self.newpick = True
                elif self.left == 2:
                    self.ready2 = False
                    self.newpick = False 
                self.drawDots(0)
                self.drawDots(1)
                if self.left == 0:
                    self.newpick = True

    def blendImages(self):
        self.delaunay = Delaunay(self.startPoints)
        alpha = float(self.horizontalSlider.value()/20)
        if len(self.startImage.shape) == 2:
            startnew = np.array(Image.new("L", (self.startImage.shape[1], self.startImage.shape[0]), "white"))
            endnew = np.array(Image.new("L", (self.startImage.shape[1], self.startImage.shape[0]), "white"))
            final = np.array(Image.new("L", (self.startImage.shape[1], self.startImage.shape[0]), "white"))
        if len(self.startImage.shape) == 3:
            startnew = np.array(Image.new("RGB", (self.startImage.shape[1], self.startImage.shape[0]), "white"))
            endnew = np.array(Image.new("RGB", (self.startImage.shape[1], self.startImage.shape[0]), "white"))
            final = np.array(Image.new("RGB", (self.startImage.shape[1], self.startImage.shape[0]), "white"))
        for i in self.delaunay.simplices:
            # Source
            x0s = np.float64(self.startPoints[i[0]][0])
            y0s = np.float64(self.startPoints[i[0]][1])
            x1s = np.float64(self.startPoints[i[1]][0])
            y1s = np.float64(self.startPoints[i[1]][1])
            x2s = np.float64(self.startPoints[i[2]][0])
            y2s = np.float64(self.startPoints[i[2]][1])
            # Destination
            x0d = np.float64((1 - alpha) * self.startPoints[i[0]][0] + alpha * self.endPoints[i[0]][0])
            y0d = np.float64((1 - alpha) * self.startPoints[i[0]][1] + alpha * self.endPoints[i[0]][1])
            x1d = np.float64((1 - alpha) * self.startPoints[i[1]][0] + alpha * self.endPoints[i[1]][0])
            y1d = np.float64((1 - alpha) * self.startPoints[i[1]][1] + alpha * self.endPoints[i[1]][1])
            x2d = np.float64((1 - alpha) * self.startPoints[i[2]][0] + alpha * self.endPoints[i[2]][0])
            y2d = np.float64((1 - alpha) * self.startPoints[i[2]][1] + alpha * self.endPoints[i[2]][1])
            source = np.array([[x0s, y0s], [x1s, y1s], [x2s, y2s]])
            destination = np.array([[x0d, y0d], [x1d, y1d], [x2d, y2d]])
            A = np.array([[source[0][0], source[0][1], 1, 0, 0, 0],
                        [0, 0, 0, source[0][0], source[0][1], 1],
                        [source[1][0], source[1][1], 1, 0, 0, 0],
                        [0, 0, 0, source[1][0], source[1][1], 1],
                        [source[2][0], source[2][1], 1, 0, 0, 0],
                        [0, 0, 0, source[2][0], source[2][1], 1]])
            b = np.array([[destination[0][0]],
                        [destination[0][1]],
                        [destination[1][0]],
                        [destination[1][1]],
                        [destination[2][0]],
                        [destination[2][1]]])
            h = np.linalg.solve(A, b)
            H = np.array([[h[0][0], h[1][0], h[2][0]],
                        [h[3][0], h[4][0], h[5][0]],
                        [0, 0, 1]])
            Hp = np.linalg.inv(H)
            sourceImage = self.startImage
            destinationImage = startnew
            img = Image.new('L', (sourceImage.shape[1], sourceImage.shape[0]), 0)
            ImageDraw.Draw(img).polygon([(destination[0][0], destination[0][1]), (destination[1][0], destination[1][1]), (destination[2][0], destination[2][1])], outline=255, fill=255)
            mask = np.array(img)
            nonzero = np.nonzero(mask)
            xcoords = list(nonzero[0])
            ycoords = list(nonzero[1])
            addup = np.vstack((ycoords, xcoords, np.ones((len(xcoords)))))
            finpt = Hp.dot(addup)
            if len(self.startImage.shape) == 2:
                destinationImage[xcoords, ycoords] = ndimage.map_coordinates(sourceImage, [finpt[1], finpt[0]], order=1, mode='nearest')
            elif len(self.startImage.shape) == 3:
                zeros = [0] * len(xcoords)
                ones = [1] * len(xcoords)
                twos = [2] * len(xcoords)
                destinationImage[xcoords, ycoords, zeros] = ndimage.map_coordinates(sourceImage[:,:,0], [finpt[1], finpt[0]], order=1, mode='nearest')
                destinationImage[xcoords, ycoords, ones] = ndimage.map_coordinates(sourceImage[:,:,1], [finpt[1], finpt[0]], order=1, mode='nearest')
                destinationImage[xcoords, ycoords, twos] = ndimage.map_coordinates(sourceImage[:,:,2], [finpt[1], finpt[0]], order=1, mode='nearest') 
            # Source
            x0s = np.float64(self.endPoints[i[0]][0])
            y0s = np.float64(self.endPoints[i[0]][1])
            x1s = np.float64(self.endPoints[i[1]][0])
            y1s = np.float64(self.endPoints[i[1]][1])
            x2s = np.float64(self.endPoints[i[2]][0])
            y2s = np.float64(self.endPoints[i[2]][1])
            # Destination
            x0d = np.float64((1 - alpha) * self.startPoints[i[0]][0] + alpha * self.endPoints[i[0]][0])
            y0d = np.float64((1 - alpha) * self.startPoints[i[0]][1] + alpha * self.endPoints[i[0]][1])
            x1d = np.float64((1 - alpha) * self.startPoints[i[1]][0] + alpha * self.endPoints[i[1]][0])
            y1d = np.float64((1 - alpha) * self.startPoints[i[1]][1] + alpha * self.endPoints[i[1]][1])
            x2d = np.float64((1 - alpha) * self.startPoints[i[2]][0] + alpha * self.endPoints[i[2]][0])
            y2d = np.float64((1 - alpha) * self.startPoints[i[2]][1] + alpha * self.endPoints[i[2]][1])
            source = np.array([[x0s, y0s], [x1s, y1s], [x2s, y2s]])
            destination = np.array([[x0d, y0d], [x1d, y1d], [x2d, y2d]])
            A = np.array([[source[0][0], source[0][1], 1, 0, 0, 0],
                        [0, 0, 0, source[0][0], source[0][1], 1],
                        [source[1][0], source[1][1], 1, 0, 0, 0],
                        [0, 0, 0, source[1][0], source[1][1], 1],
                        [source[2][0], source[2][1], 1, 0, 0, 0],
                        [0, 0, 0, source[2][0], source[2][1], 1]])
            b = np.array([[destination[0][0]],
                        [destination[0][1]],
                        [destination[1][0]],
                        [destination[1][1]],
                        [destination[2][0]],
                        [destination[2][1]]])
            h = np.linalg.solve(A, b)
            H = np.array([[h[0][0], h[1][0], h[2][0]],
                        [h[3][0], h[4][0], h[5][0]],
                        [0, 0, 1]])
            Hp = np.linalg.inv(H)
            sourceImage = self.endImage
            destinationImage = endnew
            img = Image.new('L', (sourceImage.shape[1], sourceImage.shape[0]), 0)
            ImageDraw.Draw(img).polygon([(destination[0][0], destination[0][1]), (destination[1][0], destination[1][1]), (destination[2][0], destination[2][1])], outline=255, fill=255)
            mask = np.array(img)
            nonzero = np.nonzero(mask)
            xcoords = list(nonzero[0])
            ycoords = list(nonzero[1])
            addup = np.vstack((ycoords, xcoords, np.ones((len(xcoords)))))
            finpt = Hp.dot(addup)
            if len(self.startImage.shape) == 2:
                destinationImage[xcoords, ycoords] = ndimage.map_coordinates(sourceImage, [finpt[1], finpt[0]], order=1, mode='nearest')
            elif len(self.startImage.shape) == 3:
                zeros = [0] * len(xcoords)
                ones = [1] * len(xcoords)
                twos = [2] * len(xcoords)
                destinationImage[xcoords, ycoords, zeros] = ndimage.map_coordinates(sourceImage[:,:,0], [finpt[1], finpt[0]], order=1, mode='nearest')
                destinationImage[xcoords, ycoords, ones] = ndimage.map_coordinates(sourceImage[:,:,1], [finpt[1], finpt[0]], order=1, mode='nearest')
                destinationImage[xcoords, ycoords, twos] = ndimage.map_coordinates(sourceImage[:,:,2], [finpt[1], finpt[0]], order=1, mode='nearest')
        # Mix Pixel
        final[np.arange(self.startImage.shape[0])] = (1 - alpha) * startnew[np.arange(self.startImage.shape[0])] + alpha * endnew[np.arange(self.startImage.shape[0])]
        imageio.imsave('temp.jpg', final)
        label = QLabel(self.graphicsView_3)
        pm = QPixmap('temp.jpg')
        pm = pm.scaled(400,300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pm)
        label.show()
        os.remove('temp.jpg')
    
    def displayAlpha(self):
        self.lineEdit.setText(str(float(self.horizontalSlider.value()/20))) 
        self.lineEdit.setAlignment(Qt.AlignCenter)

    def loadDataFromFileS(self, filePath):
        self.startImagePath = filePath
        self.startImage = imageio.imread(filePath)
        if os.path.isfile(filePath + '.txt'):
            self.mode = 0
            self.startPoints = np.loadtxt(filePath + '.txt')
            self.delaunay = Delaunay(self.startPoints)
            fig = plt.figure()
            ax = plt.axes([0, 0, 1, 1])
            ax.xaxis.set_ticks_position('top')
            ax.invert_yaxis()
            if len(self.startImage.shape) == 2:
                ax.imshow(self.startImage, extent=[0, self.startImage.shape[1], self.startImage.shape[0], 0], cmap=gray)
            else:
                ax.imshow(self.startImage, extent=[0, self.startImage.shape[1], self.startImage.shape[0], 0])
            ax.plot(self.startPoints[:,0], self.startPoints[:,1], 'o', color='red', markeredgecolor='red')
            if self.reload == True:
                ax.plot(self.startPoints[:self.nop,0], self.startPoints[:self.nop,1], 'o', color='red', markeredgecolor='red')
                ax.plot(self.startPoints[self.nop:,0], self.startPoints[self.nop:,1], 'o', color='blue', markeredgecolor='blue')
            ax.set_axis_off()
            plt.savefig('temp.png')
            plt.close()
            self.startLabel = QLabel(self.graphicsView_1)
            pm = QPixmap('temp.png')
            pm = pm.scaled(400,300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.startLabel.setPixmap(pm)
            self.startLabel.show()
            self.startLabel.mousePressEvent = self.mcEventS
            os.remove('temp.png')
        else:
            self.mode = 1
            f = open(filePath + '.txt', 'w')
            f.close()
            self.startLabel = QLabel(self.graphicsView_1)
            pm = QPixmap(filePath)
            pm = pm.scaled(400,300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.startLabel.setPixmap(pm)
            self.startLabel.show()
            self.startLabel.mousePressEvent = self.mcEventS
        if self.endImagePath != None:
            self.lineEdit.setEnabled(True)
            self.btnBlend.setEnabled(True)
            self.checkBox.setEnabled(True)
            self.horizontalSlider.setEnabled(True)

    def loadDataFromFileE(self, filePath):
        self.endImagePath = filePath
        self.endImage = imageio.imread(filePath)
        if os.path.isfile(filePath + '.txt'):
            self.mode = 0
            self.endPoints = np.loadtxt(filePath + '.txt')
            fig = plt.figure()
            ax = plt.axes([0, 0, 1, 1])
            ax.xaxis.set_ticks_position('top')
            ax.invert_yaxis()
            if len(self.endImage.shape) == 2:
                ax.imshow(self.endImage, extent=[0, self.endImage.shape[1], self.endImage.shape[0], 0], cmap=gray)
            else:
                ax.imshow(self.endImage, extent=[0, self.endImage.shape[1], self.endImage.shape[0], 0])
            ax.plot(self.endPoints[:,0], self.endPoints[:,1], 'o', color='red', markeredgecolor='red')
            if self.reload == True:
                ax.plot(self.endPoints[:self.nop,0], self.endPoints[:self.nop,1], 'o', color='red', markeredgecolor='red')
                ax.plot(self.endPoints[self.nop:,0], self.endPoints[self.nop:,1], 'o', color='blue', markeredgecolor='blue')
            ax.set_axis_off()
            plt.savefig('temp.png')
            plt.close()
            self.endLabel = QLabel(self.graphicsView_2)
            pm = QPixmap('temp.png')
            pm = pm.scaled(400,300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.endLabel.setPixmap(pm)
            self.endLabel.show()
            self.endLabel.mousePressEvent = self.mcEventE
            os.remove('temp.png')
        else:
            self.mode = 1
            f = open(filePath + '.txt', 'w')
            f.close()
            self.endLabel = QLabel(self.graphicsView_2)
            pm = QPixmap(filePath)
            pm = pm.scaled(400,300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.endLabel.setPixmap(pm)
            self.endLabel.show()
            self.endLabel.mousePressEvent = self.mcEventE
        if self.startImagePath != None:
            self.lineEdit.setEnabled(True)
            self.btnBlend.setEnabled(True)
            self.checkBox.setEnabled(True)
            self.horizontalSlider.setEnabled(True)

    def loadDataS(self):
        filePath, _ = QFileDialog.getOpenFileName(self, caption='Open jpg file ...', filter="JPG, PNG files (*.jpg *.png)")
        if not filePath:
            return
        # Member variables
        self.ready1 = False
        self.ready2 = False
        self.mode = None
        self.tempxs = None
        self.tempys = None
        self.tempxe = None
        self.tempye = None
        self.newpick = False
        self.stupid = False
        self.blockS = False
        self.blockE = True
        self.left = 0
        self.deleteone = False
        self.deletetwo = False
        self.special = False
        self.reload = False
        if os.path.isfile(filePath + '.txt'):
            with open(filePath + '.txt', 'r') as myFile:
                if self.nop == None:
                    self.nop = len(myFile.readlines())
                else:
                    self.reload = True
            with open(filePath + '.txt', 'a') as myFile:
                myFile.write('\n')
        self.loadDataFromFileS(filePath)

    def loadDataE(self):
        filePath, _ = QFileDialog.getOpenFileName(self, caption='Open jpg file ...', filter="JPG, PNG files(*.jpg *.png)")
        if not filePath:
            return
        # Member variables
        self.ready1 = False
        self.ready2 = False
        self.mode = None
        self.tempxs = None
        self.tempys = None
        self.tempxe = None
        self.tempye = None
        self.newpick = False
        self.stupid = False
        self.blockS = False
        self.blockE = True
        self.left = 0
        self.deleteone = False
        self.deletetwo = False
        self.special = False
        if os.path.isfile(filePath + '.txt'):
            with open(filePath + '.txt', 'a') as myFile:
                myFile.write('\n')
        self.loadDataFromFileE(filePath)

if __name__ == "__main__":
    currentApp = QApplication(sys.argv)
    currentForm = MorphConsumer()

    currentForm.show()
    currentApp.exec_()
