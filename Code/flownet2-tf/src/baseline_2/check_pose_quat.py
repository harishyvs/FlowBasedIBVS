import tf.transformations as tft
import math
import sys
import numpy as np

vel = np.genfromtxt('/home/harish/RRC/ICRA_2019/baseline_2/aaaaa/New_method/x_y_z_translations/baseline_2_velocities_single_1.txt', delimiter=' ')
pose = np.genfromtxt('/home/harish/RRC/ICRA_2019/baseline_2/aaaaa/New_method/x_y_z_translations/baseline_2_exp_indi.txt',delimiter=' ')
pose1 = np.zeros_like(pose)
pose1[0,:] = pose[0,:]

for i in range(pose.shape[0]-1):
    ti = pose1[i,:3]
    qi = pose1[i,3:]

    #vt = np.array([vel[i,0]*(-0.1),vel[i,1]*(-0.0988),vel[i,2]*(0.01)])
    #wt = np.array([vel[i,3]*1,vel[i,4]*(-1),vel[i,5]*2])

    vt = np.array([vel[i,0],vel[i,1],vel[i,2]])
    wt = np.array([vel[i,3],vel[i,4],vel[i,5]])


    td = vt
    qd = tft.quaternion_from_euler(wt[0],wt[1],wt[2],'sxyz')

    qi1 = tft.quaternion_multiply(qd, qi) #quaternion_multiply(q_rot, q_orig)

    #rotating vector around quat equivivalent of ri*Td(1:3,4)+ti
    tdh = np.hstack([td,0])
    ti1 = tft.quaternion_multiply(tft.quaternion_multiply(qi, tdh), tft.quaternion_conjugate(qi))[:3] + ti

    pose1[i+1,:] = np.hstack([ti1,qi1])

np.savetxt('/home/yvsharish/working/aaaaa_2/baseline_2_after_change.txt', pose1,delimiter=' ')
