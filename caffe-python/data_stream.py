import numpy as np
import os, sys
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize
from PIL import Image,ImageFilter
import caffe
import pdb

def preprocess_img(image_path,mean_image_path):
	# Before feeding the image to the network, we need to preprocess it:
    # 1) Resize image to (256, 256)
    # 2) Swap channels from RGB to BGR (for CaffeNet)
    # 3) Reshape from (H, W, K) to (K, H, W)
    # 4) Subtract ImageNet mean
    # 5) Crop or resize to (227, 227)

    img = imread(image_path)
    mean = np.load(mean_image_path)
    if img.shape!=(227,227): #transform image if not 256x256
    	# Resize image to same size as mean
        H_mean, W_mean = mean.shape[1:]
        img = imresize(img, (H_mean, W_mean))
        # Swap channels from RGB to BGR
        try:
            img = img[:, :, (2, 0, 1)]
            # Reshape from (H, W, K) to (K, H, W)
            img = img.transpose(2, 0, 1)
        except: 
            num_files = num_files-1
            pass

        # Subtract mean
        img = img - mean

        # Crop to input size of network
        H_in, W_in = net.blobs['data'].data.shape[2:]
        H0 = (H_mean - H_in) / 2
        H1 = H0 + H_in
        W0 = (W_mean - W_in) / 2
        W1 = W0 + W_in
        img = img[:, H0:H1, W0:W1]
    return img

computer_screenshots_dir = '/Users/anandsampat/project_231n/data/unorganized_raw_data/realtime'
all_files = os.listdir(computer_screenshots_dir)

tasks = ['calendar','email','programming','social_media','word_processing','web_shopping',
           'finance','news','music_editing','picture_editing','video_editing','watching_video','web_browsing','web_forum']

# Load caffe Net in Test mode.
model_file = '../caffe/models/Screenshots-transfer-hybridCNN-mini-large-fc8fc7conv3/deploy.prototxt'
weights_file = '../caffe/models/Screenshots-transfer-hybridCNN-mini-large-fc8fc7conv3/Screenshots-transfer-hybridCNN-mini-large-train_iter_10000.caffemodel'
net = caffe.Net(model_file, weights_file, caffe.TEST)

# Take in images from the "data stream" - i.e. from timestamped images (ordered by time)
all_files.pop(0)

# Use window of 4 to subset the images of interest 
for ind,f in enumerate(all_files):
	meanpath = os.path.join('/Users/anandsampat/project_231n/data/unorganized_raw_data/','hybridCNN_mean.npy')
	if ind==0: # no averaging
		files = [os.path.join(computer_screenshots_dir,f)]
	elif ind==1: # average two images
		files = [os.path.join(computer_screenshots_dir,all_files[ind-1]),os.path.join(computer_screenshots_dir,f)]
	elif ind==2:
		files = [os.path.join(computer_screenshots_dir,all_files[ind-2]),os.path.join(computer_screenshots_dir,all_files[ind-1]),os.path.join(computer_screenshots_dir,f)]
	else: 
		files = [os.path.join(computer_screenshots_dir,all_files[ind-3]),os.path.join(computer_screenshots_dir,all_files[ind-2]),os.path.join(computer_screenshots_dir,all_files[ind-1]),os.path.join(computer_screenshots_dir,f)]
	print files
	
	# Classify each image and average the final probabilities

	img = [preprocess_img(f_i,meanpath) for f_i in files]
	probs_avg = np.zeros(14)
	for i in img: 
		net.blobs['data'].data[0] = i
		net.forward()
		probs = net.blobs['prob'].data[0]
		probs_avg += probs.squeeze()
	probs_avg=probs_avg/len(img)
	print "Average probs classification: " + tasks[np.argmax(probs_avg)]

	# Average the images and find final probabilities in classifier
	N=len(files)
	img_avg=np.zeros((3,227,227),np.float)
	for i in img:
		img_avg=img_avg+i
	img_avg = img_avg/N
	img_avg=np.array(np.round(img_avg),dtype=np.uint8)
	# Use output as input to net
	net.blobs['data'].data[0] = img_avg 
	net.forward()
	probs = net.blobs['prob'].data[0]
	print "Average image classification: " + tasks[np.argmax(probs)]
	# Show average image
	plt.imshow(img_avg.transpose(1,2,0).astype('uint8'))
	plt.show()


	# Show original image
	plt.imshow(img[-1].transpose(1,2,0).astype('uint8'))
	plt.show()


# 	# Assess to test if correct class

    # word_arr = os.path.splitext(f)[0].split('_')
    # if len(word_arr)>2:
    #     c = word_arr[0]+'_'+word_arr[1]
    # else: 
    #     c = word_arr[0]
# #     print tasks[probs.argmax()]
#     if tasks[probs.argmax()] == c: 
#         test_accuracy[i] += 1
#         print test_accuracy[i]
#     counter += 1
# test_accuracy[i] = float(test_accuracy[i])/num_files
# print test_accuracy[i]
# print np.mean(test_accuracy)


