import transformations_tf as tft
import math
import sys
import numpy as np

import time
with open("../aaaaa/baseline_2_velocities_single_1_new.txt") as f:
    content = f.readlines()
# print(content)
time11=time.time()
harish=0
content = [x.strip() for x in content]
mat = []
for line in content:
    s = line.split(' ')
    # print(s)
    if  len(s) == 6:
        mat.append(s)
print(mat)
Vx=float(mat[harish][0])
Vy=float(mat[harish][1])
Vz=float(mat[harish][2])
Wx=float(mat[harish][3])
Wy=float(mat[harish][4])
Wz=float(mat[harish][5])
Vx=0.01*Vx
Vy=Vy*0.01
Vz=Vz*-0.01

Wx=0.01*Wx
Wy=0.01*Wy
Wz= -0.01*Wz

vt = np.array([Vx,Vy,Vz])
wt = np.array([Wx,Wy,Wz])
time22=time.time()

with open("../aaaaa/baseline_2_exp_indi.txt") as f:
    content = f.readlines()
time1=time.time()
harish=0
content = [x.strip() for x in content]
mat = []
for line in content:
    s = line.split(' ')
    if  len(s) == 7:
        mat.append(s)
Tx=float(mat[harish][0])
Ty=float(mat[harish][1])
Tz=float(mat[harish][2])
qx=float(mat[harish][3])
qy=float(mat[harish][4])
qz=float(mat[harish][5])
qw=float(mat[harish][6])

ti = np.array([Tx,Ty,Tz])
qi = np.array([qx,qy,qz,qw])

td = vt
qd = tft.quaternion_from_euler(wt[0],wt[1],wt[2],'sxyz')
qi1 = tft.quaternion_multiply(qd, qi) #quaternion_multiply(q_rot, q_orig)

#rotating vector around quat equivivalent of ri*Td(1:3,4)+ti
tdh = np.hstack([td,0])
ti1 = tft.quaternion_multiply(tft.quaternion_multiply(qi, tdh), tft.quaternion_conjugate(qi))[:3] + ti
time2=time.time()
timeHOMO=time2-time1+time22-time11
ftime=open("../aaaaa/baseline_2_time_iterations.txt","a+")
ftime.write("HOMO%.20f\n"%(timeHOMO))
ftime.close()
f_1=open("../aaaaa/baseline_2_after_change.txt","w")
f_1.write("%0.8f %0.8f %.8f %e %e %e %e\n" %(ti1[0],ti1[1],ti1[2],qi1[0],qi1[1],qi1[2],qi1[3]))
f_1.close()
