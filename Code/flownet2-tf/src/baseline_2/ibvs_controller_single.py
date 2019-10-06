import numpy as np
from interactionMatrix import interactionMatrix
import scipy.io
from PIL import Image
import numpy as np
import os, sys, numpy as np
import sys
from numpy import linalg as LA

def readFlow(name):
    if name.endswith('.pfm') or name.endswith('.PFM'):
        return readPFM(name)[0][:,:,0:2]

    f = open(name, 'rb')

    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')

    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()

    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))

    return flow.astype(np.float32)

# harish=readFlow('out.flo')
# print(harish)

def ibvs_controller(s,Z,cam,error):
    #here centroid mean the image coordinates of intersted point in the form (number, 2)
    #cam is the numpy interaction matrix
    # here Z matrix should be taken from the ground truth images or should be proportional to the velocities of the points , Z - of the size #image

    w=0.15
    w=0.15*2
    w=0.5*0.15
    Lsd=interactionMatrix(s,cam,Z)
    # print(Lsd.shape)
    # print(error.shape)
    
    vc=-w*np.matmul(np.linalg.pinv(Lsd),error)
    # error
    return vc


max_frames=15
total_frames=0
# while total_frames < max_frames-1:
f=open("/scratch/yvsharish/working/aaaaa/baseline_2_velocities_single_1.txt","a+")
file_in=open("/scratch/yvsharish/working/aaaaa/baseline_2_velocities_single_1_new.txt","w")

f1=open("/scratch/yvsharish/working/aaaaa/baseline_2_velocities_norm_1.txt","a+")
f2=open("/scratch/yvsharish/working/aaaaa/baseline_2.txt","w")
print('before getting flo')
error= readFlow('/home/yvsharish/test/output_dir/output_img.flo' )
print('after reading flo')
print(error.shape)
# print(error)
nx, ny = (512,384)
error=error.transpose(1,0,2)
error=flatten()
error=np.reshape(error,(nx*ny*2,-1))
# error=np.reshape(error,(512*384*2,1))
# nx, ny = (512,384)
x = np.linspace(0, nx-1, nx)
y = np.linspace(0, ny-1, ny)
s=np.array([])
for i in range(nx):
    for j in range(ny):
        s=np.hstack((s,np.array([i,j])))
print(s)
# print(sys.argv[1])
# harish=("/home/harish/RRC/ICRA_2019/habitat/habitat-sim/image/test.depth.%05.png")
if(sys.argv[1]=='0'):
    print("inside depth = 0")
    harish="/home/yvsharish/working/habitat-sim/image_baseline_2/test.depth." + str(sys.argv[1]).zfill(5) +".png"
else:
    harish="/scratch/yvsharish/working/habitat-sim/image_baseline_2_output/test.depth." + str(sys.argv[1]).zfill(5) +".png"
# harish="/home/harish/RRC/ICRA_2019/habitat-sim/image_baseline_2/test.depth.00000"+".png"
# harish="/home/harish/RRC/ICRA_2019/habitat-sim/image_baseline_2/test.depth.00019"+".png"
# print(harish)
im=Image.open(harish)
Z=(np.array(im))
Z=Z.astype('float64')


cam=np.asarray([[nx/2,0,nx/2],[0,ny/2, ny/2 ],[ 0, 0, 1]])
# print(error)

print('before ibvs_controller')
vc=ibvs_controller(s,Z,cam,error)
print('after ibvs_controller')
f1.write("%.20f\n"%(LA.norm(vc[0:3,0])))
f1.close()
f2.write("%.20f\n"%(LA.norm(vc[0:3,0])))
f2.close()
# print(vc[2,0])
# f.write("%e %e %.20f %.20f %.20f %.20f\n" %(vc[0,0],-vc[1,0],-vc[2,0],vc[3,0],vc[4,0],vc[5,0]))
# file_in.write("%e %e %.20f %.20f %.20f %.20f\n" %(vc[0,0],-vc[1,0],-vc[2,0],vc[3,0],vc[4,0],vc[5,0]))
f.write("%e %e %.20f %.20f %.20f %.20f\n" %(-vc[0,0],vc[1,0],-vc[2,0],-vc[3,0],vc[4,0],-vc[5,0]))
file_in.write("%e %e %.20f %.20f %.20f %.20f\n" %(-vc[0,0],vc[1,0],-vc[2,0],-vc[3,0],vc[4,0],-vc[5,0]))
file_in.close()
f.close()
total_frames=total_frames+1
