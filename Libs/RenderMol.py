import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as pa
import matplotlib.lines as mlines
from matplotlib.collections import PatchCollection

from NiceColours import *

ZColList={ 1:"White%95", 6:"Black", 7:"Blue", 8:"Red",
           26:"Grey",
           2:"Cyan",10:"Cyan",18:"Cyan",36:"Cyan",
           3:"Lavendar",11:"Lavendar",19:"Lavendar",37:"Lavendar",
           9:"Green",17:"Green|Green|Olive",
           35:"Green|Olive|Olive",53:"Olive", }
def GetAtomCol(Z):
    if Z in ZColList:
        C=ZColList[Z]
    elif (Z in range(22,33)) or (Z in range(40,52)):
        C="Grey"
    else:
        C="Coral"

    CL=C.split("|")
    rgb=[0.,0.,0.]
    for CN in CL:
        T=CN.split("%")
        if len(T)>1:
            CID=T[0]
            S=float(int(T[1]))/100.
        else:
            CID=T[0]
            S=1.0
        C=NiceColour(ID=CID)
        rgb[0]+=C[0]*S
        rgb[1]+=C[1]*S
        rgb[2]+=C[2]*S
    rgb=[X/len(CL) for X in rgb]

    return rgb

def lenvec(X):
    return np.sqrt(X[0]**2 + X[1]**2 + X[2]**2)

def normvec(C):
    return C/lenvec(C)


def NiceCircle(ax, x, y, radius=1.0, color=(0,0,0), zorder=None):
    R,c=radius,color
    ax.add_patch(pa.Circle((x,y), radius=R, color=c, zorder=zorder))
    a=0.3
    cp=( 1. - a*(1.-c[0]), 1. - a*(1.-c[1]), 1. - a*(1.-c[2]) )
    ax.add_patch(pa.Circle((x-0.3*R,y+0.3*R), radius=R*0.35,
                           color=cp, zorder=zorder+1))
    ax.add_patch(pa.Circle((x-0.3*R,y+0.3*R), radius=R*0.20,
                           color=(1.,1.,1.), zorder=zorder+2))

def DoProj(X, Proj, aspect=0., GetS=False):
    Y=np.dot(X, Proj.T)

    if len(X.shape)<2:
        S=np.exp(-aspect*Y[2])
        Y[0]=Y[0]*S
        Y[1]=Y[1]*S
        Y[2]=S
    else:
        S=np.exp(-aspect*Y[:,2])
        Y[:,0]=Y[:,0]*S
        Y[:,1]=Y[:,1]*S

    if GetS:
        return Y,S
    else:
        return Y

def RenderMol(ax, Z,R, Proj=None, Bonds=None,
              PlotValues=None, PHeight=None,
              PCols=("r-", "g--", "c-."),
              aspect=0.03):
    # Guess a projection
    if Proj is None or len(Proj)<=4:
        if len(Proj)==4:
            a,b,c,d=Proj
        elif len(Proj)==3:
            a,b,d=Proj
            c=a
        else:
            a,b,c,d=0,1,0,2

        a1=normvec(R[b,:]-R[a,:])
        if (len(Z)>2):
            a2x=normvec(R[d,:]-R[c,:])
        else:
            a2x=normvec(np.array([1.,1.,1.]))
        if np.dot(a1,a2x)<1e-3:
            a2x=normvec(np.array([1.,2.,3.]))

        a2=normvec(a2x - np.dot(a2x,a1)*a1)
        a3=np.cross(a1,a2)

        Proj=np.array([a1,a2,a3])

    ### Project and center the molecule
    RExtent=np.zeros((3,2))
    for k in range(3):
        RExtent[k,0]=R[:,k].min()
        RExtent[k,1]=R[:,k].max()
        R[:,k]-=(RExtent[k,0]+RExtent[k,1])/2.

    PP,SP=DoProj(R, Proj, aspect, GetS=True)
    Extent=np.zeros((3,2))
    for k in range(2):
        Extent[k,0]=PP[:,k].min()
        Extent[k,1]=PP[:,k].max()
        PP[:,k]-=(Extent[k,0]+Extent[k,1])/2.
    SP=np.exp(-aspect*PP[:,2])

    for A in range(len(Z)):
        for Ap in range(len(Z)):
            BC2=(1.5/0.53)**2
            RA=R[A,:]
            RAp=R[Ap,:]
            L2=np.sum( (RA-RAp)**2 )

            P1=PP[A ,0:2]
            P2=PP[Ap,0:2]

            angle=np.arctan2(P2[1]-P1[1],P2[0]-P1[0])
            Depth=(PP[A ,2] + PP[Ap,2])/2.
            Width=np.sqrt(np.sum((P1-P2)**2))

            if not(Bonds is None):
                DrawBond=Bonds[A,Ap]>0.9
            else:
                DrawBond=(L2<BC2) # Crudely defines bonds as <1.5 Ang
                
                
            if DrawBond and (Ap>A):
                # Has bond
                Height=0.1

                P0=P1-Height/2.*np.array([-np.sin(angle), np.cos(angle)])
                
                ax.add_patch(pa.Rectangle(
                        P0,Width,Height,angle=angle*180./np.pi,
                        color=(0.,0.,0.),                        
                        zorder=int(100*Depth-3)))

            if not(PlotValues is None) and ((A,Ap) in PlotValues):
                Data=PlotValues[(A,Ap)]
                x=Data[0]
                x-=x[0]
                sx=Width/x[-1]

                sy=2.0

                s=np.sin(angle)
                c=np.cos(angle)
                
                acut=np.pi*0.25
                if np.abs(s)>(3.*np.abs(c)):
                    if (-s*sy)<0.: sy=-sy
                else:
                    if ( c*sy)<0.: sy=-sy

                for p in range(1,len(Data)):
                    y=np.minimum(Data[p],1.2)

                    xp=P1[0]+(c*sx*x - s*sy*y)
                    yp=P1[1]+(s*sx*x + c*sy*y)
                
                    if (p==1):
                        ax.add_patch(pa.Rectangle(
                                P1,Width,sy,angle=angle*180./np.pi,
                                color=(0.9,0.9,0.9), alpha=0.2,
                                zorder=1000))
                    plt.plot(xp,yp, PCols[p-1], linewidth=4, zorder=1000)

    RP=PP[:,0:2]
    ZP=PP[:,2]
    for A in range(len(Z)):
        NiceCircle( ax, RP[A,0],RP[A,1], zorder=int(100*SP[A]),
                    radius=0.3*SP[A], color=GetAtomCol(Z[A]) )
        #pp = pa.Circle((RP[A,0],RP[A,1]), radius=0.05, transform=ax.transData)
        #ax.add_patch(pp)

    ax.axis('equal')
    X1,X2=ax.get_xlim()
    Y1,Y2=ax.get_ylim()
    Scale = 1.1 # Zoom out by 10%
    if not(PHeight is None):
        Scale=float(PHeight)/float(Y2-Y1)
        print "# Scaling from %.1f to %.1f using %.2f"%(Y2-Y1,PHeight,Scale)
    XC=(X1+X2)/2.
    XR=(X2-X1)/2. * Scale
    YC=(Y1+Y2)/2.
    YR=(Y2-Y1)/2. * Scale

    ax.set_xlim([XC-XR,XC+XR])
    ax.set_ylim([YC-YR,YC+YR])
    ax.axis('off')



