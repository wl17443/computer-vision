import cv2 as cv
import numpy as np 
from matplotlib import pyplot as plt

def imread(filename,index):
    img = cv.imread(filename,index)
    return img

def imshow(windowname,img):
    cv.imshow(windowname,img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def namedWindow(windowname, flag):
    cv.namedWindow(windowname, flag)
    cv.imshow(windowname,img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def imwrite(filename,img):
    cv.imwrite(filename,img)

def imsave(filename,index,windowname,newfilename):
    img = imread(filename,index)
    cv.imshow(windowname,img)
    k = cv.waitKey(0)
    if k == 27:
        cv.destroyAllWindows()
    elif k == ord('s'):
        imwrite(newfilename,img)
        cv.destroyAllWindows()

def pltimshow(filename,index):
    img = imread(filename,index,color,interp)
    plt.imshow(img, cmap=color,interpolation=interp)
    plt.xticks([]), plt.yticks([]) #to hide tick values on X and Y axis
    plt.show()


