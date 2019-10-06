import numpy as np
from PIL import Image
import sys
import os
# os.chdir("/home/harish/RRC/ICRA_2019/habitat-sim/examples/")
# hey=128
# command="python example_me.py --width "+ str(hey) +" --height 128  --scene /home/harish/RRC/ICRA_2019/habitat/habitat-sim/data_1/gibson/Hillsdale.glb  --max_frames 20 --save_png --depth_sensor"
# print(command)
# os.system(command)
def mse_(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])

	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err
# print(sys.argv[0],sys.argv[1])
h=int(sys.argv[2])
h=h-1
harish2="/home/yvsharish/working/habitat-sim/image_baseline_2/test.rgba." + str(h).zfill(5) +".png"
im1=Image.open(harish2)
# im1.show()
Z1=(np.asarray(im1))
harish1="/scratch/yvsharish/working/habitat-sim/image_baseline_2_output/test.rgba." + str(sys.argv[1]).zfill(5) +".png"
im2=Image.open(harish1)
# im2.shape
Z2=(np.asarray(im2))
# print(Z2[19,20,3])
# print(Z1.shape[2])
# print(Z2.shape)
err=mse_(Z1,Z2)
# print(err)
# print((Z1-Z2).shape)
# mse = (np.square(Z1 - Z2)).mean(axis=None)
# print(mse)
f2=open("/scratch/yvsharish/working/aaaaa/baseline_2_photo.txt","w")
f2.write("%f\n"%(err))
f2.close()
f3=open("/scratch/yvsharish/working/aaaaa/baseline_2_photo_all.txt","a+")
f3.write("%f\n"%(err))
f3.close()
