# import the necessary packages
from skimage import exposure
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize
import os,sys

# small_test_img_dir = '/Users/anandsampat/project_231n/data/organized_data/small_test/'
# large_test_img_dir = '/Users/anandsampat/project_231n/data/organized_data/large_test/'
# for f in os.listdir(small_test_img_dir):
# from PIL import Image,ImageFilter
# img = Image.open('/Users/anandsampat/project_231n/data/organized_data/small_test/programming_106.JPEG','r')
# gray=img.convert('L') #makes it greyscale
# grayedged = gray.filter(ImageFilter.FIND_EDGES)
# grayedged.show()
# grayedged = np.asarray(grayedged)
# gray = np.asarray(gray)
# img = np.asarray(img)
# grayedged = cv2.Canny(gray, 0, 200)
# grayedged = cv2.Laplacian(gray,cv2.CV_64F)
# 
# Image.fromarray(np.uint8(grayedged)).show()
#img = cv2.imread('/Users/anandsampat/project_231n/data/organized_data/small_test/programming_80.JPEG')
# swap channels to BGR
#img = img[:, :, (2, 0, 1)]
# gray = cv2.imread('/Users/anandsampat/project_231n/caffe/data/screenshotsdb/small_test/calendar_727.JPEG',0)

# load the query image, compute the ratio of the old height
# to the new height, clone it, and resize it

img = cv2.imread('/Users/anandsampat/project_231n/data/organized_data/small_test/programming_80.JPEG')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(1,1),1000)
# blur = cv2.bilateralFilter(gray, 11, 17, 17)
flag, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)

# ratio = grayedged.shape[0] / 300.0
# grayedged = imresize(grayedged, (300,int(grayedged.shape[1]/ratio)))
# img = imresize(img,(300,int(img.shape[1]/ratio)))

# find contours in the edged image, keep only the largest
# ones, and initialize our screen contour
(cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
screenCnt = None

# loop over our contours
max_area = 0
for c in cnts:
    # approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
 	print approx
	# if our approximated contour has four points, then
	# we can assume that we have found our screen
	if len(approx) == 4:
		# if (np.abs((approx[1][0]-approx[0][0])[0])<5) and (np.abs((approx[3][0]-approx[0][0])[1])<5):
		# 	area = (approx[1][0]-approx[0][0])[1]*(approx[3][0]-approx[0][0])[0]
		# 	if area > max_area:
		# 		max_area=area
		# 		screenCnt = approx
		screenCnt = approx
		print screenCnt
		cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)
		cv2.imshow("largest contour...", img)
		cv2.waitKey(0)