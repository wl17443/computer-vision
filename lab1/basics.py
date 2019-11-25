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

def pltimshow(filename,index,color,interp):
    img = imread(filename,index)
    plt.imshow(img, cmap=color,interpolation=interp)
    plt.xticks([]), plt.yticks([]) #to hide tick values on X and Y axis
    plt.show()

# Changing Colour-space
# BGR <-> Gray and BGR <-> HSV
# cv.COLOR_BGR2GRAY and cv.COLOR_BGR2HSV
# HSV H[0,179] S[0,255] V[0,155]

# Object Tracking
# Easier to represent a colour in HSV than BGR colour-space 
#   Take a frame
#   Convert from BGR to HSV 
#   Threshold the HSV image for a range of blue colour 
#   Extract the blue object alone 

def trackObject(filename,index):
    img = imread(filename,index)
    # Convert BGR to HSV 
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # define range of blue color in HSV 
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    # threshold the HSV image to get only blue colors 
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    # bitwise-AND mask and orifinal image 
    res = cv.bitwise_and(img, img, mask=mask)

    cv.imshow('image',img)
    cv.imshow('mask',mask)
    cv.imshow('res',res)
    cv.waitKey(0)
    
# Image thresholding 
# Simple thresholding:
#   if pixel value is greater than a threshold value, it is assigned one value
#       else it is assigned another value 
#   using cv.threshold()
#   first argument is the source image in GRAYSCALE
#   second argument is the threshold value which is used to classify the pixel
#       values 
#   third argument is the maxVal which represents the value to be given if pixel
#       value is more than the threshold value 
#   fourth argument -> differnet styles of thresholding 
#   THRESH_BINARY, THRESH_BINARY_INV, THRESH_TRUNC, THRESH_TOZERO, THRESH_TOZERO_INV 

def simpleThresholding(filename, index):
    img = imread(filename, index)
    ret,thresh1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
    ret,thresh2 = cv.threshold(img,127,255,cv.THRESH_BINARY_INV)
    ret,thresh3 = cv.threshold(img,127,255,cv.THRESH_TRUNC)
    ret,thresh4 = cv.threshold(img,127,255,cv.THRESH_TOZERO)
    ret,thresh5 = cv.threshold(img,127,255,cv.THRESH_TOZERO_INV)

    titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
    images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

    for i in range(6):
        plt.subplot(2,3,i+1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    
    plt.show()

# Adaptive thresholding 
#   algorithm calculates the threshodl for a small region of the image 
#   3 special input parameters and only one output argument 
#       1. adaptive method 
#           ADAPTIVE_THRESH_MEAN_C -> threshold value is the mean of the neighbouring area
#           ADAPTIVE_THRESH_GAUSSIAN_C -> threshold value if the weited sum 
#               of neighbourhood values where weights are a gaussian window 
#       2. block size - size of neighbourhood area 
#       3. C - constant which is subtracted from the mean or weighted mean calculated 

def adaptiveThresholding(filename,index):
    img = imread(filename, index)
    img = cv.medianBlur(img,5)

    ret,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
    th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)

    titles = ['Original Image', 'Global Thresholding (v=127)', 'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    images = [img, th1, th2, th3]

    for i in range(4):
        plt.subplot(2,2,i+1), plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()

# Otsu's binarization 

def otsuBin(filename,index):
    img = imread(filename,index)
    # global thresholding
    ret1,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
    # otsu's thresholding
    ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    # otsu's thresholding after Guassian filtering
    blur = cv.GaussianBlur(img,(5,5),0)
    ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

    images = [img, 0, th1, 
              img, 0, th2, 
              blur, 0, th3]
    titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
            'Original Noisy Image','Histogram', "Otsu's Thresholding",
            'Gaussian Filtered Image','Histogram',"Otsu's Thresholding"]

    for i in range(3):
        plt.subplot(3,3,i*3+1), plt.imshow(images[i*3],'gray')
        plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
        plt.subplot(3,3,i*3+2), plt.hist(images[i*3].ravel(),256)
        plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
        plt.subplot(3,3,i*3+3), plt.imshow(images[i*3+2],'gray')
        plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])

    plt.show()

otsuBin('mandrill.jpg',0)
