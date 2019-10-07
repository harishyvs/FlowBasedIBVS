import numpy as np

def interactionMatrix(s,cam,Z):
    #Z has dimensions of image
    p=512
    q=384
    KK=cam;
    px=KK[0,0]
    py=KK[1,1]
    v0=KK[0,2]
    u0=KK[1,2]
    Lsd=np.zeros((p*q*2,6))

    for m in range(0,p*q*2-2,2):
        x=(int(s[m])-int(u0))/px
        y=(int(s[m+1])-int(v0))/py
        t=int(s[m])
        u=int(s[m+1])
        print(Z.shape)
        print(Z[u,t])
        Zinv=10/(Z[u,t]+1)
        print(Zinv)

        Lsd[m,0]=-Zinv
        Lsd[m,1]=0
        Lsd[m,2]=x*Zinv
        Lsd[m,3]=x*y
        Lsd[m,4]=-(1+x**2)
        Lsd[m,5]=y


        Lsd[m+1,0]= 0
        Lsd[m+1,1]= -Zinv
        Lsd[m+1,2]= y*Zinv
        Lsd[m+1,3]= 1+y**2
        Lsd[m+1,4]= -x*y
        Lsd[m+1,5]= -x
    return Lsd
