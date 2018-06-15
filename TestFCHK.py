import numpy as np
import matplotlib.pyplot as plt

from ReadFCHK import *
from GTOGaussian import *
from ConcurrenceLib import *

from QuickElements import *
from NiceColours import *

QE=QuickElements()

HaToAng=0.53
AngToHa=1./0.53

D=FCHKFile("Data/Benzene.fchk")

ZAtom=D['Nuclear charges']
IDAtom=[QE.GetID(Z) for Z in ZAtom]

NAtom=len(ZAtom)
RAtom=D['Current cartesian coordinates'].reshape(NAtom,3)*HaToAng

Basis=GTOGaussian(D)
Orbs=Orbitals(D)

# Temp
for k in range(Orbs.NH2, Orbs.NL2+1):
    T=Orbs.phiCoeffs[k,:]
    print "%3d %.4f"%(k,np.sum(T**2))
    for b in range(len(T)):
        if np.abs(T[b])>0.1: # Select more than 10% contribution
            gb,l,m=Basis.GetGaussianProps(b)
            print "%6.3f %d %d %s"%(T[b],l,m,
                                    NiceTriplet(gb['Position']))

###################################

Nx=31
z0=1.5/0.53
x=np.linspace(-5.,5.,Nx)
xyz=np.zeros((Nx**2,3))
for k in range(len(x)):
    xyz[Nx*k:(Nx*k+Nx),0]=x[k]
    xyz[Nx*k:(Nx*k+Nx),1]=x
    xyz[Nx*k:(Nx*k+Nx),2]=z0

CC2=Concurrence(Orbs, Basis, xyz, W=0.3)

#Out=CC2.nL
Out=np.diag(CC2.Cs)

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


mx,my=np.meshgrid(x,x)
ax.plot_surface(mx,my,Out.reshape((Nx,Nx)),
                cmap=cm.coolwarm,
                rstride=1,cstride=1)
ax.view_init(elev=45., azim=45.)
plt.show()
quit()


""" Old debug from here """


def SetupShitGrids(NZ=201, NR=101, NT=8, ZMax=8., RMax=8.):
    Z=np.linspace(-ZMax,ZMax,NZ)
    R=np.linspace(0.,RMax,NR)+(RMax/NR)/2.
    Theta=np.arange(NT)/float(NT)*np.pi

    xyz=np.zeros((NZ,3))
    xyz[:,2]=Z
    
    xyzAll=np.zeros((NR*NT*NZ,3))
    WAll=2.*np.ones((NR*NT*NZ,))*(Z[1]-Z[0])*(R[1]-R[0])*(Theta[1]-Theta[0])
    for kr in range(NR):
        for t in range(NT):
            for kz in range(NZ):
                indx=kr*NT*NZ + t*NZ + kz
                WAll[indx]*=R[kr]
                xyzAll[indx]\
                    =[ R[kr]*np.cos(Theta[t]),
                       R[kr]*np.sin(Theta[t]),
                       Z[kz] ]

    return xyz, xyzAll, WAll

def TestOrbitals(Basis, phiCoeffs=None, DoOrbitals=None, DoPlot=False):
    import matplotlib.pyplot as plt

    xyz, xyzAll, WAll = SetupShitGrids()

    NBasis=Basis.NBasis
    for b in range(NBasis):
        GG=0.
        GZ=0.
        if DoOrbitals is None:
            GG+=Basis.GetGaussian(b, xyzAll)
            GZ+=Basis.GetGaussian(b, xyz)
        else:
            if b<DoOrbitals:
                GG+=Basis.GetOrbGrid(phiCoeffs[b,:], xyzAll)
                GZ+=Basis.GetOrbGrid(phiCoeffs[b,:], xyz)

        gb,l,m=Basis.GetGaussianProps(b)
        Norm=np.dot(GG**2, WAll)
        print "Basis %3d with (%2d,%2d) has normalisation %.4f"%(b, l, m, Norm)
        Z=xyz[:,2]
        if DoPlot:
            plt.plot(Z, GZ)

    if DoPlot:
        plt.axis([-5.,5.,-4.,4.])
        plt.show()

# Old stuff
TestOrbitals(Basis, Orbs.phiCoeffs, DoOrbitals=33)

