#! /user/local/bin/python3.4

import os
import os.path
from pathlib import Path
import time
import numpy as np
import imageio
from scipy.interpolate import RectBivariateSpline
from scipy import ndimage
from PIL import Image, ImageDraw
from scipy.spatial import Delaunay

# Affine Class
class Affine:
    # Initializer
    def __init__(self, source, destination):
        # Input Arguments Verification
        if source.shape[0] != 3 | source.shape[1] != 2 | destination[0] != 3 | destination[1] != 2:
            raise ValueError("Input arguments do not have the expected dimensions.")
        for i in range(3):
            for j in range(2):
                if type(source[i][j]) != np.float64 or type(destination[i][j]) != np.float64:
                    raise ValueError("Input arguments are not of type float64.")
        # Initialize an instance
        self.source = source
        self.destination = destination
        # Compute for matrix
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
        self.matrix = H
        self.matrixp = np.linalg.inv(H)

    # member function: transform
    def transform(self, sourceImage, destinationImage):
        # Input Arguments Verification
        if not isinstance(sourceImage, np.ndarray):
            raise TypeError("sourceImage is not a numpy array.")
        if not isinstance(destinationImage, np.ndarray):
            raise TypeError("destinationImage is not a numpy array.")
        img = Image.new('L', (sourceImage.shape[1], sourceImage.shape[0]), 0)
        ImageDraw.Draw(img).polygon([(self.destination[0][0], self.destination[0][1]), (self.destination[1][0], self.destination[1][1]), (self.destination[2][0], self.destination[2][1])], outline=255, fill=255)
        mask = np.array(img)
        nonzero = np.nonzero(mask)
        xcoords = list(nonzero[0])
        ycoords = list(nonzero[1])
        addup = np.vstack((ycoords, xcoords, np.ones((len(xcoords)))))
        finpt = self.matrixp.dot(addup)
        destinationImage[xcoords, ycoords] = ndimage.map_coordinates(sourceImage, [finpt[1], finpt[0]], order=1, mode='nearest')

# Blender Class
class Blender:
    # Initializer
    def __init__(self, startImage, startPoints, endImage, endPoints):
        # Input Arguments Verification
        if not isinstance(startImage, np.ndarray):
            raise ValueError("startImage is not an instance of numpy array.")
        if not isinstance(startPoints, np.ndarray):
            raise ValueError("startPoints is not an instance of numpy array.")
        if not isinstance(endImage, np.ndarray):
            raise ValueError("endImage is not an instance of numpy array.")
        if not isinstance(endPoints, np.ndarray):
            raise ValueError("endPoints is not an instance of numpy array.")
        # Initialize an instance
        self.startImage = startImage
        self.startPoints = startPoints
        self.endImage = endImage
        self.endPoints = endPoints
        self.delaunay = Delaunay(startPoints)
    
    # member function: getBlendedImage
    def getBlendedImage(self, alpha):
        startnew = np.array(Image.new("L", (self.startImage.shape[1], self.startImage.shape[0]), "white"))
        endnew = np.array(Image.new("L", (self.startImage.shape[1], self.startImage.shape[0]), "white"))
        final = np.array(Image.new("L", (self.startImage.shape[1], self.startImage.shape[0]), "white"))
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
            nparrs = np.array([[x0s, y0s], [x1s, y1s], [x2s, y2s]])
            nparrd = np.array([[x0d, y0d], [x1d, y1d], [x2d, y2d]])
            aff = Affine(nparrs, nparrd)
            aff.transform(self.startImage, startnew)
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
            nparrs = np.array([[x0s, y0s], [x1s, y1s], [x2s, y2s]])
            nparrd = np.array([[x0d, y0d], [x1d, y1d], [x2d, y2d]])
            aff = Affine(nparrs, nparrd)
            aff.transform(self.endImage, endnew)
        # Mix Pixel
        final[np.arange(self.startImage.shape[0])] = (1 - alpha) * startnew[np.arange(self.startImage.shape[0])] + alpha * endnew[np.arange(self.startImage.shape[0])]
        
        return final

    def generateMorphVideo(self, targetFolderPath, sequenceLength, includeReversed=True):
        # If directory exists
        if Path(targetFolderPath).is_dir():
            pass
        # If directory doesn't exist
        else:
            os.mkdir(targetFolderPath)
        savepath = targetFolderPath + '/morph.mp4'
        videomaker = imageio.get_writer(savepath, fps=5)
        temp_frames = []
        filenumb = 1
        for alpha in np.linspace(0, 1, sequenceLength):
            frame = self.getBlendedImage(alpha)
            videomaker.append_data(frame)
            temp_frames.append(frame)
            filename = targetFolderPath + '/frame{:03d}.jpg'.format(filenumb)
            imageio.imsave(filename,frame)
            filenumb += 1
        if includeReversed == True:
            for frame in reversed(temp_frames):
                videomaker.append_data(frame)
                filename = targetFolderPath + '/frame{:03d}.jpg'.format(filenumb)
                imageio.imsave(filename,frame)
                filenumb += 1
        videomaker.close()

# ColorAffine Class
class ColorAffine:
    # Initializer
    def __init__(self, source, destination):
        # Input Arguments Verification
        if source.shape[0] != 3 | source.shape[1] != 2 | destination[0] != 3 | destination[1] != 2:
            raise ValueError("Input arguments do not have the expected dimensions.")
        for i in range(3):
            for j in range(2):
                if type(source[i][j]) != np.float64 or type(destination[i][j]) != np.float64:
                    raise ValueError("Input arguments are not of type float64.")
        # Initialize an instance
        self.source = source
        self.destination = destination
        # Compute for matrix
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
        self.matrix = H
        # Compute for inverse matrix
        self.matrixp = np.linalg.inv(H)

    # member function: transform
    def transform(self, sourceImage, destinationImage):
        # Input Arguments Verification
        if not isinstance(sourceImage, np.ndarray):
            raise TypeError("sourceImage is not a numpy array.")
        if not isinstance(destinationImage, np.ndarray):
            raise TypeError("destinationImage is not a numpy array.")
        # Compute for mask
        img = Image.new('L', (sourceImage.shape[1], sourceImage.shape[0]), 0)
        ImageDraw.Draw(img).polygon([(self.destination[0][0], self.destination[0][1]), (self.destination[1][0], self.destination[1][1]), (self.destination[2][0], self.destination[2][1])], outline=255, fill=255)
        mask = np.array(img)
        nonzero = np.nonzero(mask)
        # Extract pixel color value
        xcoords = list(nonzero[0])
        ycoords = list(nonzero[1])
        addup = np.vstack((ycoords, xcoords, np.ones((len(xcoords)))))
        finpt = self.matrixp.dot(addup)
        zeros = [0] * len(xcoords)
        ones = [1] * len(xcoords)
        twos = [2] * len(xcoords)
        destinationImage[xcoords, ycoords, zeros] = ndimage.map_coordinates(sourceImage[:,:,0], [finpt[1], finpt[0]], order=1, mode='nearest')
        destinationImage[xcoords, ycoords, ones] = ndimage.map_coordinates(sourceImage[:,:,1], [finpt[1], finpt[0]], order=1, mode='nearest')
        destinationImage[xcoords, ycoords, twos] = ndimage.map_coordinates(sourceImage[:,:,2], [finpt[1], finpt[0]], order=1, mode='nearest')

# ColorBlender Class
class ColorBlender:
    # Initializer
    def __init__(self, startImage, startPoints, endImage, endPoints):
        # Input Arguments Verification
        if not isinstance(startImage, np.ndarray):
            raise ValueError("startImage is not an instance of numpy array.")
        if not isinstance(startPoints, np.ndarray):
            raise ValueError("startPoints is not an instance of numpy array.")
        if not isinstance(endImage, np.ndarray):
            raise ValueError("endImage is not an instance of numpy array.")
        if not isinstance(endPoints, np.ndarray):
            raise ValueError("endPoints is not an instance of numpy array.")
        # Initialize an instance
        self.startImage = startImage
        self.startPoints = startPoints
        self.endImage = endImage
        self.endPoints = endPoints
        self.delaunay = Delaunay(startPoints)

    # member function: getBlendedImage
    def getBlendedImage(self, alpha):
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
            nparrs = np.array([[x0s, y0s], [x1s, y1s], [x2s, y2s]])
            nparrd = np.array([[x0d, y0d], [x1d, y1d], [x2d, y2d]])
            aff = ColorAffine(nparrs, nparrd)
            aff.transform(self.startImage, startnew)
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
            nparrs = np.array([[x0s, y0s], [x1s, y1s], [x2s, y2s]])
            nparrd = np.array([[x0d, y0d], [x1d, y1d], [x2d, y2d]])
            aff = ColorAffine(nparrs, nparrd)
            aff.transform(self.endImage, endnew)
        # Mix Pixel
        final[np.arange(self.startImage.shape[0])] = (1 - alpha) * startnew[np.arange(self.startImage.shape[0])] + alpha * endnew[np.arange(self.startImage.shape[0])]
        
        return final

    def generateMorphVideo(self, targetFolderPath, sequenceLength, includeReversed=True):
        # If directory exists
        if Path(targetFolderPath).is_dir():
            pass
        # If directory doesn't exist
        else:
            os.mkdir(targetFolderPath)
        savepath = targetFolderPath + '/morph.mp4'
        videomaker = imageio.get_writer(savepath, fps=5)
        temp_frames = []
        filenumb = 1
        for alpha in np.linspace(0, 1, sequenceLength):
            frame = self.getBlendedImage(alpha)
            videomaker.append_data(frame)
            temp_frames.append(frame)
            filename = targetFolderPath + '/frame{:03d}.jpg'.format(filenumb)
            imageio.imsave(filename,frame)
            filenumb += 1
        if includeReversed == True:
            for frame in reversed(temp_frames):
                videomaker.append_data(frame)
                filename = targetFolderPath + '/frame{:03d}.jpg'.format(filenumb)
                imageio.imsave(filename,frame)
                filenumb += 1
        videomaker.close()

# Conditional Main Block
if __name__ == "__main__":
    '''   
    # Gray Testing Code
    ims = imageio.imread('Tiger2Gray.jpg')
    imd = imageio.imread('WolfGray.jpg')
    triangle1 = np.loadtxt('tiger2.jpg.txt')
    triangle2 = np.loadtxt('wolf.jpg.txt')
    tri1 = Delaunay(triangle1)
    tri2 = Delaunay(triangle2)
    Blen = Blender(ims, triangle1, imd, triangle2)
    Blen.generateMorphVideo('video', 20, True)
    filenumb = 1
    for i in np.linspace(0, 1, 80):
        start = time.time()
        result = Blen.getBlendedImage(i)
        filename = 'myFrames/frame' + str(filenumb) + '.jpg'
        filenumb = filenumb + 1
        imageio.imsave(filename, result)
        end = time.time()
        print(end-start)
    '''
    # Color Testing Code
    ims = imageio.imread('Tiger2Color.jpg')
    imd = imageio.imread('WolfColor.jpg')
    triangle1 = np.loadtxt('tiger2.jpg.txt')
    triangle2 = np.loadtxt('wolf.jpg.txt')
    tri1 = Delaunay(triangle1)
    tri2 = Delaunay(triangle2)
    Blen = ColorBlender(ims, triangle1, imd, triangle2)
    Blen.generateMorphVideo('video', 20, True)
    '''
    filenumb = 1
    for i in np.linspace(0, 1, 80):
        start = time.time()
        result = Blen.getBlendedImage(i)
        filename = 'myFrames/frame' + str(filenumb) + '.jpg'
        filenumb = filenumb + 1
        imageio.imsave(filename, result)
        end = time.time()
        print(end-start)
    '''
