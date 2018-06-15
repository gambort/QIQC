import numpy as np
import matplotlib.pyplot as plt

from ReadFCHK import *
from GTOGaussian import *
from ConcurrenceLib import *

from QuickElements import *
from NiceColours import *

from SphGrid import *

from  optparse import OptionParser

parser = OptionParser()
parser.add_option('--ID', type="string", default="HCl-127")
parser.add_option('--Shiftx', type=float, default=0.)
parser.add_option('--Shifty', type=float, default=0.)
parser.add_option('--Shiftz', type=float, default=0.)
parser.add_option('-W', type=float, default=0.3)

parser.add_option('--NGrid', type=int, default=6)
parser.add_option('--Radius', type=float, default=0.6)

parser.add_option('--Mode', type="string", default="Cs")

parser.add_option('--Grid', type="string", default="PGaussian")

(Opts, args) = parser.parse_args()

QE=QuickElements()

class Logger:
    def __init__(self, FName=None):
        self.Init(FName)
    def Init(self, FName=None):
        if FName is None: self.F=None
        else: self.F=open(FName, "w")

    def Log(self, Str):
        print(Str)
        if not(self.F is None):
            self.F.write(Str+"\n")
    def Close(self):
        if not(self.F is None): self.F.close()


L=Logger("Data/Pairs-%s-%.1f.log"%(Opts.ID, Opts.Radius))

RayShift=np.array([Opts.Shiftx,Opts.Shifty,Opts.Shiftz])*AngToHa

AtomPairs=[(0,0),(0,1),(1,1)]
if Opts.ID in ("Benzene","Benzene-CAM","Benzene-M06L"):
    ID="Benzene"
    AtomPairs=[(0,0),(0,1),(0,2),(0,3),(0,6),(0,7),(6,6),(6,7)]
elif Opts.ID in ("Fe","Fe-CAM","Fe-M06L","Fe-Aug"):
    ID="Fe-Compound"+Opts.ID[2:]
    AtomPairs=[(0,0),(0,6),(6,6),(6,12),(12,12),(12,9),(9,9),(9,15),(15,15)]
elif Opts.ID in ("Fe2p","Fe2p-CAM","Fe2p-M06L"):
    ID="Fe2p-Compound"+Opts.ID[4:]
    AtomPairs=[(0,0),(0,6),(6,6),(6,12),(12,12),(12,9),(9,9),(9,15),(15,15)]
elif Opts.ID in ("Fe2pCO6","Fe2pCO6-CAM","Fe2pCO6-M06L"):
    ID="Fe2p-CO6"+Opts.ID[7:]
    AtomPairs=[(0,0),(0,1),(1,1),(1,2),(2,2)]
elif Opts.ID[:4]=="C2H4":
    ID=Opts.ID
    AtomPairs=[(0,0),(0,1),(1,1),(1,4),(4,4)]
elif Opts.ID in ("CO2","CO2x2"):
    ID=Opts.ID
    AtomPairs=[(0,0),(0,1),(1,1),(0,2)]
else:
    ID=Opts.ID

D=FCHKFile("Data/%s.fchk"%(ID))
ZAtom=D['Nuclear charges']
IDAtom=[QE.GetID(Z) for Z in ZAtom]

NAtom=len(ZAtom)
RAtom=D['Current cartesian coordinates'].reshape(NAtom,3)

RMin2=1e10;
for A in range(NAtom):
    for Ap in range(A+1,NAtom):
        R2=((RAtom[A,:]-RAtom[Ap,:])**2).sum()
        RMin2=min(R2,RMin2)
           

for A in range(NAtom):
    L.Log("%-3s %3d %s [Ang]"%(IDAtom[A], ZAtom[A],
                         NiceTriplet(RAtom[A,:]*HaToAng)))


# Setup basis and orbitals
Basis=GTOGaussian(D)
Orbs=Orbitals(D)


NGrid=Opts.NGrid
NGrid3=NGrid**3
if Opts.Radius>0.:
    Radius=Opts.Radius*AngToHa
else:
    Radius=-Opts.Radius*np.sqrt(RMin2)/2.

L.Log("Shortest bond:  %.4f [Ang]"%(np.sqrt(RMin2)*HaToAng))
L.Log("Gausssian rad.: %.4f [Ang]"%(Radius*HaToAng))

GridID=Opts.Grid[0:3].upper()
if GridID=="SPH":
    L.Log("Using spherical grid")
    Cube,W=GetSphGrid(R=Radius)
elif GridID=="RAN":
    from numpy.random import normal
    Cube=normal(scale=Radius,size=(NGrid,3))
    R2=Cube[:,0]**2+Cube[:,1]**2+Cube[:,2]**2
    W=np.exp(-R2/Radius**2)
    W/=np.sum(W)
else:
    x,wx,lwx=GaussHermiteWeights(NGrid,S=Radius)
    o=0.*x+1.

    L.Log("Using gaussian grid %d^3"%(NGrid))
    L.Log("Range from %.3f to %.3f"%(x[0], x[-1]))

    def outrio(a,b,c):
        X=np.outer(a,b)
        return np.einsum(X,[0,1],c,[2,],[0,1,2])

    X=outrio(x,o,o).reshape((NGrid3,))
    Y=outrio(o,x,o).reshape((NGrid3,))
    Z=outrio(o,o,x).reshape((NGrid3,))
    R2=X**2+Y**2+Z**2


    W=outrio(wx,wx,wx).reshape([NGrid3,])
    if GridID=="PGA":
        W*=( np.exp(-R2/Radius**2)
             - np.exp(-2.*R2/Radius**2) )
        W/=np.sum(W)
    elif GridID=="SGA":
        a=np.sqrt(2.)
        W*=( np.exp(-R2/Radius**2)
             - a**3*np.exp(-a**2*R2/Radius**2) )
    else:
        W*=np.exp(-R2/Radius**2)
        W/=np.sum(W)

    Cube=np.zeros((NGrid3,3))
    Cube[:,0]=X
    Cube[:,1]=Y
    Cube[:,2]=Z


for (A,A2) in AtomPairs:
    xyz=RAtom[A,:]+Cube+RayShift
    xyz2=RAtom[A2,:]+Cube+RayShift

    CC=Concurrence(Orbs, Basis, xyz, xyz2=xyz2, W=Opts.W)
    if Opts.Mode=="Cs2":
        Cs=CC.Cs2El
    elif Opts.Mode=="CsS":
        Cs=CC.CsSinglet
    else:
        Cs=CC.Cs

    CsAvg=(1.-2.*Opts.W)
    CsAvg=0.

    CsAA2=np.dot(np.dot(W, Cs), W)
    Dev=CsAA2-CsAvg


    if not(A==A2):
        CCB=Concurrence(Orbs, Basis, (xyz+xyz2)/2., W=Opts.W)
        if Opts.Mode=="Cs2":
            CsB=CCB.Cs2El
        else:
            CsB=CCB.Cs

        CsAA2B=np.dot(np.dot(W, CsB), W)
        DevB=CsAA2B-CsAvg
    else:
        DevB=Dev

    Rep,RepB=1000.*Dev, 1000.*DevB
    Rep,RepB=1000.*(0.5+Dev)/1.5, 1000.*(DevB+0.5)/1.5

    L.Log("%-3s(%2d) %-3s(%2d) %7.0f %7.0f"\
        %(IDAtom[A], A, IDAtom[A2], A2, Rep, RepB))

L.Close()
