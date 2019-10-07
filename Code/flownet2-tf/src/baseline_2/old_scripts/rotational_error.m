t0=[0,0,0];
r1=[0.996510734105524, -0.0175460023839052, 0.034149230975343, -0.0687783689435416];%desired
r2=[0.996522426605225, -0.0292984992265701, 0.0368666909635067, -0.0687428712844849];%resultant
%quaternion of form a+bi+cj+dk
r0=[1,0,0,0];
%r1=[];%desired
%r2=[];%resultant
%translational error
%rdiff=r2*quatinv(r1);
%rdiff1=r1*quatinv(r0);
%ediff=quat2eul(rdiff,'XYZ');
r11=quaternion(r1);
e1=euler(r11,'ZYX','frame');
e2=quat2eul(r2,'XYZ');
%rotational error