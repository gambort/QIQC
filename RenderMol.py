import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as pa
import matplotlib.lines as mlines
from matplotlib.collections import PatchCollection

def NiceCircle(ax, x, y, radius=1.0, color=(0,0,0)):
    R,c=radius,color
    ax.add_patch(pa.Circle((x,y), radius=R, color=c))
    a=0.3
    cp=( 1. - a*(1.-c[0]), 1. - a*(1.-c[1]), 1. - a*(1.-c[2]) )
    ax.add_patch(pa.Circle((x-0.3*R,y+0.3*R), radius=R*0.35, color=cp))
    ax.add_patch(pa.Circle((x-0.3*R,y+0.3*R), radius=R*0.20, color=(1.,1.,1.)))

def RenderMol(ax, Z,R):
    Proj=np.array([[1.,0.,0.],[0.,1.,0.]])
    RP=np.dot(R, Proj.T)

    for k in range(len(Z)):
        for kp in range(k+1,len(Z)):
            BC2=(1.5/0.53)**2
            Rk=R[k,:]
            Rkp=R[kp,:]
            L2=np.sum( (Rk-Rkp)**2 )
            if (L2<BC2):
                # Has bond
                P1=np.dot(Rk, Proj.T)
                P2=np.dot(Rkp, Proj.T)

                angle=np.arctan2(P2[1]-P1[1],P2[0]-P1[0])
                
                Width=np.sqrt(np.sum((P1-P2)**2))
                Height=0.1
                
                P0=P1-Height/2.*np.array([-np.sin(angle), np.cos(angle)])
                
                ax.add_patch(pa.Rectangle(
                        P0,Width,Height,angle=angle*180./np.pi,
                        color=(0.,0.,0.)))
                
                             
    for k in range(len(Z)):
        NiceCircle( ax, RP[k,0],RP[k,1], radius=0.5, color=(0,0,1.) )
        #pp = pa.Circle((RP[k,0],RP[k,1]), radius=0.05, transform=ax.transData)
        #ax.add_patch(pp)

    plt.axis('equal')
    plt.axis('off')
    plt.show()



