import numpy as np
import cv2
from matplotlib import pyplot as plt

#load an color image in grayscale
def read_writeimage():
    img = cv2.imread('./img/22.jpg',0)
    cv2.imwrite('./img/220.jpg',img)
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def read_writeimage1():
    img = cv2.imread('./img/22.jpg',0)
    cv2.imshow('image',img)
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('./img/220.png',img)
        cv2.destroyAllWindows() 

def read_imageplt():
    img = cv2.imread('./img/22.jpg',0)
    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

if __name__ == '__main__':
    read_writeimage()
    read_writeimage1()
    read_imageplt()
