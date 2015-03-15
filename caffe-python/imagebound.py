# import the necessary packages
from skimage import exposure
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize

img = cv2.imread('/Users/anandsampat/project_231n/caffe/data/screenshotsdb/small_test/calendar_727.JPEG')
# swap channels to BGR
img = img[:, :, (2, 0, 1)]
# gray = cv2.imread('/Users/anandsampat/project_231n/caffe/data/screenshotsdb/small_test/calendar_727.JPEG',0)

# load the query image, compute the ratio of the old height
# to the new height, clone it, and resize it
ratio = img.shape[0] / 300.0
orig = img.copy()
img = imresize(img, (300,int(img.shape[1]/ratio)))
# img = imutils.resize(img, height = 300)
 
# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray.astype('uint8'))
gray = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(gray, 30, 200)

# find contours in the edged image, keep only the largest
# ones, and initialize our screen contour
(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
screenCnt = None

# loop over our contours
for c in cnts:
    # approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.03 * peri, True)
 
	# if our approximated contour has four points, then
	# we can assume that we have found our screen
	if len(approx) == 4:
		screenCnt = approx
		break

cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)
cv2.imshow("largest contour...", img)
cv2.waitKey(0)