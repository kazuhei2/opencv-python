import sys
import numpy as np
import cv2

args = sys.argv

img = cv2.imread(args[1])

imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
imgYUV = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

imgY = imgYUV[:,:,0]

laplacian = cv2.Laplacian(imgY, cv2.CV_64F)
canny = cv2.Canny(imgY, 100, 200)
dx = cv2.Sobel(imgY, cv2.CV_64F, 1, 0, ksize=3)
dy = cv2.Sobel(imgY, cv2.CV_64F, 0, 1, ksize=3)
grad = np.sqrt(dx ** 2 + dy ** 2)

cv2.imshow('imgY', imgY)
cv2.imshow('laplacian', laplacian)
cv2.imshow('dx', dx)
cv2.imshow('dy', dy)
cv2.imshow('grad', grad)
cv2.imshow('canny', canny)

cv2.waitKey(0)

cv2.destroyAllWindows()
