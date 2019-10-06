# /home/yvsharish/working/aaaaa/baseline_2_velocities_single_1_new.txt
# /home/yvsharish/working/aaaaa/baseline_2_exp_indi.txt
# /home/yvsharish/working/aaaaa/baseline_2_after_change.txt","w"
import transformations_tf as tft
import math
import sys
import numpy as np

import time
with open("/scratch/yvsharish/working/aaaaa/baseline_2_velocities_single_1_new.txt") as f:
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
# print(Wx)
# Vx=-0.1*Vx
# Vy=Vy*-0.09883337089
# Vz=Vz* -0.09
# Vx=-0.09883337089*Vx
# Vy=Vy*-0.09883337089
# Vz=Vz*0.09

# #while giving to harit
# Vx=0.001*Vx
# Vy=Vy*0.001
# Vz=Vz*0.005
Vx=0.01*Vx
Vy=Vy*0.01
Vz=Vz*-0.01
#after giving to harit
# Vx=1*Vx
# Vy=Vy*1
# Vz=Vz*1

Wx=0.01*Wx
Wy=0.01*Wy
Wz= -0.01*Wz

vt = np.array([Vx,Vy,Vz])
wt = np.array([Wx,Wy,Wz])
time22=time.time()
#
# #while giving to harit
# # Wx=1*Wx
# # Wy=-1*Wy
# # Wz= 2*Wz
#
# #after giving to harit
# # Wx=1*Wx
# # Wy=1*Wy
# # Wz=1*Wz
#
#
# # print(Vx,Vy,Vz)
# D1=euler_matrix(Wx,Wy,Wz,'sxyz')
# # print(D1)
# D1[0:3,3]=[Vx,Vy,Vz]
#
# # print(D1)

with open("/scratch/yvsharish/working/aaaaa/baseline_2_exp_indi.txt") as f:
    content = f.readlines()
# print(content)
time1=time.time()
harish=0
content = [x.strip() for x in content]
mat = []
for line in content:
    s = line.split(' ')
    # print(s)
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
# qd = tft.quaternion_from_euler(0,0,0.5,'sxyz')
qi1 = tft.quaternion_multiply(qd, qi) #quaternion_multiply(q_rot, q_orig)
# qi1 =np.array([ 0.0087265, 0, 0, 0.9999619])#quaternion_multiply(q_rot, q_orig)

#rotating vector around quat equivivalent of ri*Td(1:3,4)+ti
tdh = np.hstack([td,0])
ti1 = tft.quaternion_multiply(tft.quaternion_multiply(qi, tdh), tft.quaternion_conjugate(qi))[:3] + ti
# ti1 = np.array([0,0,0]) + ti

# pose1 = np.zeros_like(pose)
# pose1 = np.hstack([ti1,qi1])
time2=time.time()
timeHOMO=time2-time1+time22-time11
ftime=open("/scratch/yvsharish/working/aaaaa/baseline_2_time_iterations.txt","a+")
ftime.write("HOMO%.20f\n"%(timeHOMO))
ftime.close()
f_1=open("/scratch/yvsharish/working/aaaaa/baseline_2_after_change.txt","w")
f_1.write("%0.8f %0.8f %.8f %e %e %e %e\n" %(ti1[0],ti1[1],ti1[2],qi1[0],qi1[1],qi1[2],qi1[3]))
f_1.close()
#
#
# # print(Tx)
# # qx = quaternion_about_axis(alpha, xaxis)
# # qy = quaternion_about_axis(beta, yaxis)
# # qz = quaternion_about_axis(gamma, zaxis)
# # q = quaternion_multiply(qx, qy)
# q = [qx,qy,qz,qw]
# D2 = quaternion_matrix(q)
# D2[0:3,3]=[Tx,Ty,Tz]
# # print(D2)
#
# D3=np.dot(D2,D1)
# final_quat=quaternion_from_matrix(D3)
# # print(final_quat)
# # print(D3)
# ans_x=D3[0,3]
# ans_y=D3[1,3]
# ans_z=D3[2,3]
# print(final_quat)
# print(ans_x,ans_y,ans_z)
# f_1=open("/home/harish/RRC/ICRA_2019/baseline_2/aaaaa/baseline_2_after_change.txt","w")
# f_1.write("%0.8f %0.8f %.8f %e %e %e %e\n" %(D3[0,3],D3[1,3],D3[2,3],final_quat[0],final_quat[1],final_quat[2],final_quat[3]))
# f_1.close()
# # D3[0:3,3]=[0,0,0]
# # final_quat=quaternion_from_matrix(D3)
# # print(final_quat)
