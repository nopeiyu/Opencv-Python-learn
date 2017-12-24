import numpy as np
import cv2

#load an color image in grayscale
img = cv2.imread('./img/22.jpg',0)
cv2.imwrite('./img/220.jpg',img)
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
