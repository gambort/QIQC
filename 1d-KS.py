#!/sw/python/2.7.6/bin/python


import sys
sys.path.append("C:/Software/Python/WFT/1D-Old/")

import numpy as np
from numpy import sqrt,exp
from math import ceil, floor
from scipy.interpolate import spline

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib import cm


def orthogonal_proj(zfront, zback):
    a = (zfront+zback)/(zfront-zback)
    b = -2*(zfront*zback)/(zfront-zback)
    # -0.0001 added for numerical stability as suggested in:
    # http://stackoverflow.com/questions/23840756
    return np.array([[1,0,0,0],
                     [0,1,0,0],
                     [0,0,a,b],
                     [0,0,-0.0001,zback]])
proj3d.persp_transformation = orthogonal_proj


from BasicRoutines import *
from HamRoutines import *
from KSRoutines import *

from  optparse import OptionParser

import os.path
import yaml

def FindMin(x,y):
    i0=np.argmin(y)
    print i0
    if (i0==0):
        ii=range(0,3)
    elif (i0==(len(x)-1)):
        ii=range(len(x)-3,len(x))
    else:
        ii=range(i0-1,i0+2)
    
    p=np.polyfit(x[ii],y[ii],2)
    x0=p[1]/(-2.0*p[0])
    if (x0<np.min(x[ii])) or (x0>np.max(x[ii])):
        x0=x[i0]
    y0=np.polyval(p, x0)

    print x0,y0
    return x0,y0




parser = OptionParser()
parser.add_option('-R', type=float, default=4.0)
parser.add_option('-W', type=float, default=0.3)
parser.add_option('-Z', type=float, default=1.0)
parser.add_option('-E', type=float, default=0.0)

parser.add_option('--PlotType', type=int, default=1)
parser.add_option('--WriteData', type=int, default=0)


parser.add_option('--Show', action="store_true", default=False)

(Opts, args) = parser.parse_args()

NEig=6

h=0.25
XM=8.0

R=Opts.R
dVMax=1 # Set larger if R>6
Z1=1.0 
Z2=Opts.Z # Can be used to break symmetry, set larger than Z1

Weight=Opts.W # f=2[1-Weight,Weight] - weight of KS excited state

ZR=[ (Z1,-0.5*R), (Z2,0.5*R) ]

print "R=%.2f, Z1=%.2f, Z2=%.2f, W=%.2f"%(R,Z1,Z2,Weight)

# Set up the system
Sys,x=SetupSys(h=h, XM=XM, Order=4, ZR=ZR)

Data2DName="Data/GS2_%04.1f_%03.1f_%03.1f.yaml"%(R,Z1,Z2)

if (os.path.exists(Data2DName)) and (Sys['h']==0.25):
    f=file(Data2DName, 'r')
    D=yaml.load(f)
    f.close()
    
    E2, Phi2 = D['E2'], D['Phi2']
else:
    # Do a two electron calculation and save
    E2,Phi2=Solve2D(Sys, NEig)
    if (Sys['h']==0.25):
        f=file(Data2DName, 'w')
        yaml.dump({'E2':E2, 'Phi2':Phi2}, f)
        f.close()

# Get the density
BasPGS=KSProps2D(Sys, E2, Phi2, f=[1.0-Weight,Weight] )


print "N. Weights:", BasPGS['FN'][0:2]

W0=BasPGS['FN'][1]/2.0


Suf="_%03.1f_%03.1f_%04.1f_%.2f-All"%(Z1,Z2,R,Weight)



fW=[2.0-Weight,Weight]
# Invert the KS system assuming double occupancy of each shell

v0=0
SysKS,KSPGS=KSInv1D(Sys, BasPGS['n'], f=fW, ZL=-1,
                    ShowIter=50, dVMax=dVMax, v0=v0,
                    MaxIter=500,
                    ErrBreak=1e-5 )

Vs=SysKS['V']

E0=BasPGS['EE']
T=BasPGS['T']
EExt=BasPGS['EExt']
Ts=KSPGS['Ts']
EHxc=E0-Ts-EExt
ts=KSPGS['tk'][0:2]
    
print "E0   = %.4f, T    = %.4f, Ts   = %.4f, EHxc = %.4f" \
    %(E0, T, Ts, EHxc)


n=BasPGS['n']

n20=BasPGS['n2k'][0]
n21=BasPGS['n2k'][1]

phi=KSPGS['phik']
n2s0=2.*np.outer(phi[:,0], phi[:,0])**2
n2s1=( np.outer(phi[:,0], phi[:,1]) 
       - np.outer(phi[:,1], phi[:,0]) )**2


print h**2*np.sum(np.sum(n20))
print h**2*np.sum(np.sum(n21))
print h**2*np.sum(np.sum(n2s0))
print h**2*np.sum(np.sum(n2s1))

Nx=len(Vs)

i0=np.argmax( np.abs(x-R/2. + 1.0) )
n0=n[i0]

PlotType=Opts.PlotType
fs=16

if False:
    fig = plt.figure()
    fig.set_size_inches(10.,5.,forward=True)
    plt.plot(x, n, linewidth=4, label="Ens.")
    plt.plot(x, BasPGS['nk'][:,0], linewidth=4, label="GS")
    plt.plot(x, BasPGS['nk'][:,1], linewidth=4, label="Ex.")
    plt.axis([-8.,8.,0.,1.2])
    plt.legend(loc="upper right")
    plt.xlabel("$x$")
    plt.ylabel("$n(x)$")
    ImgName="Images/1DDens"+Suf+".png"
    if Opts.Show:
        plt.show()
    else:
        plt.savefig(ImgName, dpi=150)
    plt.clf()


if (PlotType==1):
    XCut=4.0

    ge=n21/n20-1.
    ges=n2s1/n2s0-1.

    w=Weight

    #p=1.-(4.*w*(1.+ge))/(3.*(1.+w*ge))
    p=1.-4./3.*(w*n21)/( (1.-w)*n20 + w*n21 )
    #C=np.maximum( (3.*p-1.)/2., 0 )
    C=np.maximum( ((1.-w)*n20-w*n21)/((1.-w)*n20+w*n21), 0. )

    #ps=1.-(4.*w*(1.+ges))/(2.*(1.+w*ges))
    ps=1.-4./3.*(w*n2s1)/( (1.-w)*n2s0 + w*n2s1 )
    #Cs=np.maximum( (3.*ps-1.)/2., 0 )
    Cs=np.maximum( ((1.-w)*n2s0-w*n2s1)/((1.-w)*n2s0+w*n2s1), 0. )

    
    fig = plt.figure()
    fig.set_size_inches(10.,5.,forward=True)
    
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    ax1.set_position([0.0,0.03,0.5,0.97])
    ax2.set_position([0.5,0.03,0.5,0.97])

    ax1.view_init(elev=45., azim=45.)
    ax2.view_init(elev=45., azim=45.)

    ii=np.abs(x)<=XCut
    C=C[ii,:][:,ii]
    Cs=Cs[ii,:][:,ii]
    X,Xp=np.meshgrid(x[ii],x[ii])

    ax1.plot_surface(X,Xp,C,rstride=1,cstride=1,
                     cmap=cm.RdYlBu,linewidth=0)
    ax2.plot_surface(X,Xp,Cs,rstride=1,cstride=1,
                     cmap=cm.RdYlBu,linewidth=0)

    Axes=(ax1,ax2)
    Lbls=("$C(x,x')$","$C_s(x,x')$")
    for k in (0,1):
        ax=Axes[k]
        zL=Lbls[k]
        ax.set_xlim3d([-XCut,XCut])
        ax.set_ylim3d([-XCut,XCut])
        ax.set_zlim3d([0.,1.])
        ax.set_xlabel("$x'$", fontsize=fs)
        ax.set_ylabel("$x$", fontsize=fs)
        ax.text2D(0.05,0.8, zL, fontsize=fs,
                  transform=ax.transAxes)
        ax.plot([-R/2],[-R/2.],[1.01],"ok",
                markersize=7.*np.sqrt(Z1))
        ax.plot([R/2],[R/2.],[1.01],"ok",
                markersize=7.*np.sqrt(Z2))

    ImgName="Images/1DConc"+Suf+".png"
    if Opts.Show:
        plt.show()
    else:
        plt.savefig(ImgName, dpi=150)
    #plt.show()
elif (PlotType==2):
    ge=n21/n20-1.
    ges=n2s1/n2s0-1.

    w=Weight

    #p=1.-(4.*w*(1.+ge))/(3.*(1.+w*ge))
    p=1.-4./3.*(w*n21)/( (1.-w)*n20 + w*n21 )
    C=np.maximum( (3.*p-1.)/2., 0 )

    #ps=1.-(4.*w*(1.+ges))/(2.*(1.+w*ges))
    ps=1.-4./3.*(w*n2s1)/( (1.-w)*n2s0 + w*n2s1 )
    Cs=np.maximum( (3.*ps-1.)/2., 0 )

    n2=(1.-w)*n20 + w*n21
    n2s=(1.-w)*n2s0 + w*n2s1

    G=n2/np.outer(n, n)
    Gs=n2s/np.outer(n, n)


    fig = plt.figure()
    fig.set_size_inches(10.,5.,forward=True)
    
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    ax1.set_position([0.0,0.03,0.5,0.97])
    ax2.set_position([0.5,0.03,0.5,0.97])

    ax1.view_init(30,30)
    ax2.view_init(30,30)

    X,Xp=np.meshgrid(x,x)
    ax1.plot_surface(X,Xp,G,rstride=1,cstride=1,
                     cmap=cm.RdYlBu,linewidth=0)
    ax2.plot_surface(X,Xp,Gs,rstride=1,cstride=1,
                     cmap=cm.RdYlBu,linewidth=0)

    Axes=(ax1,ax2)
    Lbls=("$C(x,x')$","$C_s(x,x')$")
    for k in (0,1):
        ax=Axes[k]
        zL=Lbls[k]
        ax.set_xlim3d([-12.5,12.5])
        ax.set_ylim3d([-12.5,12.5])
        ax.set_zlim3d([0.,1.])
        ax.set_xlabel("$x'$", fontsize=fs)
        ax.set_ylabel("$x$", fontsize=fs)
        ax.text2D(0.05,0.8, zL, fontsize=fs,
                  transform=ax.transAxes)    

    ImgName="Images/1DG"+Suf+".png"
    if Opts.Show:
        plt.show()
    else:
        plt.savefig(ImgName, dpi=150)
    #plt.show()
else:
    n0*=2.
    plt.plot(x,n[:], ":", lw=1)
    plt.plot(x[i0],[0.],"x")
    plt.plot(x,n20[:,i0]/n0, "-", lw=3, label="$h_{2,0}$")
    plt.plot(x,n21[:,i0]/n0, "-", lw=3, label="$h_{2,1}$")
    plt.plot(x,n2s0[:,i0]/n0, "--", lw=3, label="$h_{2,s0}$")
    plt.plot(x,n2s1[:,i0]/n0, "--", lw=3, label="$h_{2,s1}$")

    plt.axis([-10.,10.,0.,1.])
    plt.legend(loc="upper right")
    
    plt.show()
