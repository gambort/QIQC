#!/sw/python/2.7.6/bin/python


import sys
sys.path.append("C:/Software/Python/WFT/1D/")

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


parser = OptionParser()
parser.add_option('-R', type=float, default=4.0)
parser.add_option('-Z', type=float, default=1.0)
parser.add_option('-T', type=float, default=-1.0)

parser.add_option('--PlotType', type=int, default=1)
parser.add_option('--WriteData', type=int, default=0)


parser.add_option('--Show', action="store_true", default=False)

(Opts, args) = parser.parse_args()

NEig=64

h=0.25
XM=8.0

R=Opts.R
dVMax=1 # Set larger if R>6
Z1=1.0 
Z2=Opts.Z # Can be used to break symmetry, set larger than Z1

ZR=[ (Z1,-0.5*R), (Z2,0.5*R) ]

print "R=%.2f, Z1=%.2f, Z2=%.2f, T=%.0f"%(R,Z1,Z2,Opts.T)

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
BasPGS=KSProps2D(Sys, E2, Phi2, f=[1.0,])

n=BasPGS['n']
x=BasPGS['x']


WaveFunctions=[]
for k in range(NEig):
    Psik=np.reshape( Phi2[:,k],(Sys['N'],Sys['N']) )
    AS=np.sum(np.diag(Psik**2))*Sys['h']
    Singlet=(AS>1e-8)
    Degen=1
    if not(Singlet): Degen=3
    WaveFunctions.append( { 'Psi':Psik,
                            'E':E2[k],
                            'Degen': Degen,
                            'Singlet':Singlet,
                            'Triplet':not(Singlet),
                            } )
    print "| %-22s"%(" \Psi_%02d : %6.3f, %d"%(k, E2[k], Degen)),
    if (k%3)==2:
        print "|"
print

def GetW(WFns, kbT=0.1):
    f=np.zeros( (len(WFns),) )
    D=np.zeros( (len(WFns),) )
    for k in range(len(WFns)):
        f[k]=np.exp(-(WFns[k]['E']-WFns[0]['E'])/kbT)
        D[k]=WFns[k]['Degen']
    return f/np.sum(f*D),D

def GetCs(WaveFunctions, T):
    W,DW=GetW(WaveFunctions, T)
    print "Occ[0,1,2,last 5] = %.3f %.3f %.3f %.3f"\
        %(W[0],W[1],W[2],(W[-5:]*DW[-5:]).sum())

    A=0.
    B=0.
    F=0.
    for k in range(NEig):
        Wfn=WaveFunctions[k]
        P2=Wfn['Psi']**2
        if Wfn['Singlet']:
            B+=0.5*W[k]*P2
            F+=0.5*W[k]*P2
        else:
            A+=W[k]*P2
            B+=0.5*W[k]*P2
            F-=0.5*W[k]*P2

    return np.maximum(0, (F-A)/(A+B))

if Opts.T>=0:
    Suff="%04.1f_%.1f_%03.0f"%(Opts.R,Opts.Z,Opts.T)
else:  
    Suff="%04.1f_%.1f_All"%(Opts.R,Opts.Z)

if Opts.T>0.:
    TT=(Opts.T,)
    NT=1
else:
    TT=(1.,13.,25.,50.,100.,200.,400.,800.)
    NT=8

if NT==1:
    fig = plt.figure()
    fig.set_size_inches(5.,4.*len(TT),forward=True)
    axAll=(plt.gca(),)
else:
    fig,axAll=plt.subplots(2,4,
                           sharex=True,sharey=True,squeeze=True)
    fig.set_size_inches(16.,8.,forward=True)
    axAll=[item for sublist in axAll for item in sublist]
    for i in range(4):
        for j in range(2):
            k=i+j*4
            P0=[0.12,0.12,0.86,0.86]
            P1=[P0[0]/4. + i/4.,P0[1]/2. + (1-j)/2.,P0[2]/4.,P0[3]/2.]
            axAll[k].set_position(P1)



for k in range(NT):
    Cs=GetCs(WaveFunctions, TT[k]/1000.)
    
    XCut=4.
    ii=np.abs(x)<=XCut
    xC=x[ii]
    CsC=(Cs[ii].T)[ii]
    X,Xp=np.meshgrid(xC,xC)

    ax=axAll[k]
    surf=ax.imshow(CsC,extent=[-XCut,XCut,-XCut,XCut],
               origin='center',
               cmap=cm.RdYlBu,vmin=0.,vmax=1.)
    ax.axis('equal')

    if NT==1:
        fig.colorbar(surf, orientation='vertical',
                     ticks=[0.,0.2,0.4,0.6,0.8,1.0],
                     extend='both',
                     )
    ax.scatter([-Opts.R/2.,Opts.R/2.],[-Opts.R/2.,Opts.R/2.],
                c=(0.,0.,0.,0.5),
                s=[80.,80.*np.sqrt(Opts.Z)],
                )
    ax.text(0.03,0.97,"$\mathcal{C}(x,x')$",
            verticalalignment="top",
            fontsize=16,
            color='white',
            transform=ax.transAxes)
    ax.text(0.97,0.03,"$k_BT=%.0f$ [mHa]"%(TT[k]),
            horizontalalignment="right",
            fontsize=16,
            color='white',
            transform=ax.transAxes)

    ax.set_xlim([-XCut,XCut])
    ax.set_ylim([-XCut,XCut])
    #ax.set_position([0.12,0.12,0.86,0.86])

    if (Opts.T>0.):
        plt.xlabel("$x$", fontsize=16)
        plt.ylabel("$x'$", fontsize=16)
        plt.tight_layout()

plt.savefig("Images/1DAtT_%s.png"%(Suff))
#plt.savefig("Images/1DAtT_%s.eps"%(Suff))

plt.show()
