import numpy as np
from PIL import Image
import sys
import os
def mse_(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])

	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err
h=int(sys.argv[2])
h=h-1
harish2="../image_baseline_2/desired_image.png"
im1=Image.open(harish2)
Z1=(np.asarray(im1))
harish1="../image_baseline_2_output/test.rgba." + str(sys.argv[1]).zfill(5) +".png"
im2=Image.open(harish1)
Z2=(np.asarray(im2))
err=mse_(Z1,Z2)
f2=open("../aaaaa/baseline_2_photo.txt","w")
f2.write("%f\n"%(err))
f2.close()
f3=open("../aaaaa/baseline_2_photo_all.txt","a+")
f3.write("%f\n"%(err))
f3.close()
