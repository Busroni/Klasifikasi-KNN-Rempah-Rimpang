import numpy as np
from skimage.feature import graycomatrix
from skimage import color, img_as_ubyte
import cv2 as cv
import csv

import setupGLCM as setGLCM

# GLCM properties


class ModellingGLCM:
    def __init__(self, image):
        self.image = image

    def process(self):
        # GLCM properties
        img = cv.imread(self.image)
        # Convert to Greyscale
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        newNameImage = './static/img_proses/gray.jpg'
        cv.imwrite(newNameImage, gray)
        print("proses "+newNameImage + " selesai")
        image = img_as_ubyte(gray)
        bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144,
                         160, 176, 192, 208, 224, 240, 255])  # 16-bit
        inds = np.digitize(image, bins)
        print(inds)
        max_value = inds.max()+1
        print(np.pi)
        matrix_coocurrence = graycomatrix(inds, [1], [
            0, np.pi/4, np.pi/2, 3*np.pi/4], levels=max_value, normed=False, symmetric=False)
        setup = setGLCM.setGLCM(matrix_coocurrence)
        (con, hom, eng, corr) = (setup.contrast_feature(),
                                 setup.homogeneity_feature(), setup.correlation_feature(), setup.energy_feature())
        arr = np.hstack((con, hom, corr, eng))
        print(arr)

        return arr
