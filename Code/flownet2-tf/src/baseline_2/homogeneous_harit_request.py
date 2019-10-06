import transformations_tf as tft
import math
import sys
import numpy as np

with open("/home/yvsharish/working/aaaaa_2/baseline_2_velocities_single_1.txt") as f:
    content = f.readlines()
# print(content)
harish=int(sys.argv[1])
content = [x.strip() for x in content]
mat = []
for line in content:
    s = line.split(' ')
    # print(s)
    if  len(s) == 6:
        mat.append(s)
Vx=float(mat[harish][0])
Vy=float(mat[harish][1])
Vz=float(mat[harish][2])
Wx=float(mat[harish][3])
Wy=float(mat[harish][4])
Wz=float(mat[harish][5])

vt = np.array([Vx,Vy,Vz])
wt = np.array([Wx,Wy,Wz])

with open("/home/yvsharish/working/aaaaa_2/baseline_2_exp_indi.txt") as f:
    content = f.readlines()
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
#qi=np.array([qw,qx,qy,qz])

td = vt
qd = tft.quaternion_from_euler(wt[0],wt[1],wt[2],'sxyz')
# qd = tft.quaternion_from_euler(0,0,0.5,'sxyz')
qi1 = tft.quaternion_multiply(qd,qi)
 #quaternion_multiply(q_rot, q_orig)
#qi1=qi
# qi1 =np.array([ 0.0087265, 0, 0, 0.9999619])#quaternion_multiply(q_rot, q_orig)

#rotating vector around quat equivivalent of ri*Td(1:3,4)+ti
tdh = np.hstack([td,0])
ti1 = tft.quaternion_multiply(tft.quaternion_multiply(qi, tdh), tft.quaternion_conjugate(qi))[:3] + ti
#ti1 = np.array([0,0,0.75]) + ti
print("homogeneous - qd",qd,"qi1",qi1,"wt",wt,"qi",qi)

# pose1 = np.zeros_like(pose)
# pose1 = np.hstack([ti1,qi1])
f_1=open("/home/yvsharish/working/aaaaa_2/baseline_2_after_change.txt","a+")
f_1.write("%0.8f %0.8f %.8f %e %e %e %e\n" %(ti1[0],ti1[1],ti1[2],qi1[0],qi1[1],qi1[2],qi1[3]))
#f_1.write("%0.8f %0.8f %.8f %e %e %e %e\n" %(ti1[0],ti1[1],ti1[2],qi1[1],qi1[2],qi1[3],qi1[0]))

f_1.close()
