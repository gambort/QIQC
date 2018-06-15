import numpy as np
import matplotlib.pyplot as plt

from ReadFCHK import *
from GTOGaussian import *
from ConcurrenceLib import *

from QuickElements import *
from NiceColours import *

import sys
from  optparse import OptionParser

parser = OptionParser()

parser.add_option('--ListFile', type="string", default=None)

parser.add_option('--ID', type="string", default="C2H4_0")
parser.add_option('--Shiftz', type=float, default=0.0)
parser.add_option('--Range', type=float, default=-1.)
parser.add_option('--Centre', default=False, action="store_true")
parser.add_option('-W', type=float, default=0.0)

parser.add_option('--BondAxes', type="string", default=None)
parser.add_option('--BondPos', type="string", default=None)
parser.add_option('--K', type=int, default=0)

parser.add_option('--Suff', type="string", default=None)
parser.add_option('--PlotType', type="string", default="Cs")

parser.add_option('--Show', default=False, action="store_true")
parser.add_option('--SaveData', default=False, action="store_true")
(Opts, args) = parser.parse_args()


CsAvg, DataRange=0., [0.,250.,500.,750.,1000.]
#CsAvg, DataRange=1., [-300.,-200.,-100.,-50.,0.]

QE=QuickElements()


class DataFile:
    def __init__(self, FileName):
        self.D=FCHKFile(FileName)
        self.ZAtom=self.D['Nuclear charges']
        self.IDAtom=[QE.GetID(Z) for Z in self.ZAtom]

        self.NAtom=len(self.ZAtom)
        self.RAtom=self.D['Current cartesian coordinates']\
            .reshape(self.NAtom,3)

        self.Basis=GTOGaussian(self.D)
        self.Orbs=Orbitals(self.D)

def DoPlanarPlot(DF, Opts):
    # Do the plots using Opts
    RAtom=DF.RAtom
    NAtom=DF.NAtom
    Basis=DF.Basis
    Orbs=DF.Orbs

    def Permute(X,P): return np.dot(X, P)

    # Use bonds to form the plane
    RM,PU=GetPU(RAtom)
    if not(Opts.BondAxes is None) and (NAtom>2):
        B=[X.split("-") for X in Opts.BondAxes.split(",")]
        y1=RAtom[int(B[0][1]),:]-RAtom[int(B[0][0]),:]
        y2=RAtom[int(B[1][1]),:]-RAtom[int(B[1][0]),:]
        e1=y1/np.sqrt(np.dot(y1,y1))
        y2-=np.dot(y2,e1)*e1
        e2=y2/np.sqrt(np.dot(y2,y2))
        e3=np.cross(e1,e2)
        PU=np.zeros((3,3))
        PU[:,0]=e1
        PU[:,1]=e2
        PU[:,2]=e3

    # Determine the position
    if Opts.K>=0:
        K0=int(np.round(Opts.K))
        RK=RAtom[K0,:]

        K1=K0
        f=0.
    # Override if BondPos spectified "N1-N2,f"
    if not(Opts.BondPos is None):
        T=Opts.BondPos.split("-")
        K0=int(T[0])
        T=T[1].split(",")
        K1=int(T[0])
        f=float(T[1])

    RK = (1.-f)*RAtom[K0,:] + f*RAtom[K1,:]
    SuffLoc="%d-%d,%.1f"%(K0,K1,f)

    if Opts.Centre:
        RM = RK # Centre on the bond point
    
    if Opts.Range>0.:
        Range=Opts.Range*AngToHa
    else:
        TAtom=Permute(RAtom-RM, PU)
        Range=max( np.abs(TAtom[:,0]).max(), np.abs(TAtom[:,1]).max() )*1.2
        Range=np.ceil(Range*2.)/2.

    Nx=51
    z0=Opts.Shiftz
    x=np.linspace(-Range,Range,Nx)
    xyz=np.zeros((Nx**2,3))
    for i in range(len(x)):
        for j in range(len(x)):
            xyz[j*Nx+i,:]=(x[i]*PU[:,0] + x[j]*PU[:,1] + z0*PU[:,2]).T + RM

    #xyz=Permute(xyz-RM, PU)
    RAtomP=Permute(RAtom-RM, PU)
    RKP=Permute(RK-RM, PU)

    print "RM = %s"%(NiceTriplet(RM))
    print "RK = %s"%(NiceTriplet(RK))
    print "Rtl= %s"%(NiceTriplet(xyz[0,:]))
    print "Rbr= %s"%(NiceTriplet(xyz[-1,:]))
    print "a1 = %s"%(NiceTriplet(PU[:,0]))
    print "a2 = %s"%(NiceTriplet(PU[:,1]))
    print "a3 = %s"%(NiceTriplet(PU[:,2]))

    zLbl=Opts.PlotType
    if Opts.PlotType=="n":
        CC2=Concurrence(Orbs, Basis, xyz, DiagOnly=True,
                        W=Opts.W)
        Out=100.*np.log10(CC2.n.reshape((Nx,Nx)))
    elif Opts.PlotType=="nH":
        CC2=Concurrence(Orbs, Basis, xyz, DiagOnly=True,
                        W=Opts.W)
        Out=10000.*CC2.nH.reshape((Nx,Nx))
    elif Opts.PlotType=="nL":
        CC2=Concurrence(Orbs, Basis, xyz, DiagOnly=True,
                        W=Opts.W)
        Out=10000.*CC2.nL.reshape((Nx,Nx))
    elif Opts.PlotType=="nT":
        CC2=Concurrence(Orbs, Basis, xyz, DiagOnly=True,
                        W=Opts.W)
        Out=10000.*(CC2.nL-CC2.nH).reshape((Nx,Nx))
    else:
        if Opts.K<0: 
            CC2=Concurrence(Orbs, Basis, xyz, W=Opts.W, DiagOnly=True)
            Out=1000.*(CC2.Cs-CsAvg)
            Out=Out.reshape((Nx,Nx))
        else:
            CC2=Concurrence(Orbs, Basis, xyz,
                            xyz2=RK,
                            W=Opts.W)
            if Opts.PlotType=="Cs2":
                Cs=CC2.Cs2El.reshape((Nx,Nx))
            elif Opts.PlotType=="CsS":
                Cs=CC2.CsSinglet.reshape((Nx,Nx))
            else:
                Cs=CC2.Cs.reshape((Nx,Nx))

        #Out=1000.*(Cs-CsAvg)
        #zLbl="$1000(C_s-\\bar{C}_s)$"

            Out=1000.*np.maximum(Cs,0.)
            zLbl="$1000C_s$"

        #Out=1000.*(0.5+Cs)/1.5
        #zLbl="$1000(C_s+0.5)/1.5$"

        #Out=1000.*Cs**0.25
        #zLbl="$1000 C_s^{1/4}$"


        
    print "Maximum/minimum values: %.1f %.1f"%(Out.min(), Out.max())


    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    mx,my=np.meshgrid(x,x)
    surf=ax.plot_surface(mx,my,Out,
                         cmap=cm.coolwarm,
                         #cmap=cm.seismic,
                         linewidth=1, edgecolors=(0.,0.,0.,0.5),
                         vmin=DataRange[0], vmax=DataRange[-1],
                         rstride=1,cstride=1,
                         zorder=0)

    def AddText(ax, x,y,z, Text):
        ax.text3D(x,y,z,Text,
                  horizontalalignment="center",
                  verticalalignment="center",
                  clip_on=False,
                  fontsize=16, color="black")
        
    RA=RAtomP
    Z0=0.
    ZM=1005.
    for A in range(NAtom):
        ax.plot3D([RA[A,0],RA[A,0]],[RA[A,1],RA[A,1]],[Z0,ZM],
                  color=(0.,0.,0.), alpha=0.9, linewidth=3)
        
        AddText(ax, RA[A,0]-0.3,RA[A,1],ZM, DF.IDAtom[A])
        
    ax.plot3D([RKP[0],RKP[0]],[RKP[1],RKP[1]],[Z0,ZM], "--",
              color=(0.25,0.25,0.25),
              alpha=1.0, linewidth=3)

    ax.scatter(RA[:,0],RA[:,1],ZM,
               color=(0,0,0), alpha=0.9, marker='o', s=80,
               zorder=100)

    ax.view_init(elev=75., azim=45.)

    ax.set_xlim(-Range,Range)
    ax.set_ylim(-Range,Range)
    ax.set_zlim(DataRange[0], DataRange[-1])

    R=np.floor(Range*HaToAng*2.)/2.
    xyTicks=np.arange(-R,R+0.1,0.5)
    
    ax.set_xticks(xyTicks*AngToHa)
    ax.set_xticklabels(xyTicks)
    ax.set_xlabel("$x$ [$\\AA$]", fontsize=16)

    ax.set_yticks(xyTicks*AngToHa)
    ax.set_yticklabels(xyTicks)
    ax.set_ylabel("$y$ [$\\AA$]", fontsize=16)

    ax.set_zticks(DataRange)
    ax.set_title(zLbl, fontsize=16)

    cax = fig.add_subplot(111)
    cax.set_position([0.90,0.03,0.03,0.94])
    fig.colorbar(surf, orientation='vertical',
                 ticks=DataRange,
                 extend='both',
                 cax=cax
                 )
    ax.set_position([0.03,0.03,0.94,0.94])


    if Opts.Suff is None:
        Suff="_%.1f_%.1f"%(Opts.W, Opts.Shiftz)
        Suff+="_%s"%(SuffLoc)
        if not(Opts.PlotType=="Cs"):
            Suff+="_"+Opts.PlotType
    else:
        Suff="_"+Opts.Suff

    print "#"*10+" xyz file"
    print NAtom
    print Opts.ID
    for k in range(NAtom):
        print "%-3s %s"%(DF.IDAtom[k], 
                         NiceTriplet(RAtom[k],delim=None))

    if Opts.SaveData:
        np.savez("Data/Plane%s%s"%(Opts.ID,Suff),
                 x=x, Data=Out)

    if Opts.Show:
        plt.show()
    else:
        plt.savefig("Images/eps/Plane%s%s.pdf"%(Opts.ID,Suff))
        plt.savefig("Images/Plane%s%s.png"%(Opts.ID,Suff),
                    density=150)
    plt.close('all')

    return None

if Opts.ListFile is None:
    DF=DataFile("Data/%s.fchk"%(Opts.ID))
    DoPlanarPlot(DF, Opts)
else:
    LF=open(Opts.ListFile, "r")
    IDCurrent=""
    for L in LF:
        OO=L.split()
        if len(OO)==0:
            continue

        (OptsL,ArgsL)=parser.parse_args(sys.argv[1:]+OO)

        if not(OptsL.ID==IDCurrent):
            IDCurrent=OptsL.ID
            print "Loading file %s"%(IDCurrent)
            DF=DataFile("Data/%s.fchk"%(IDCurrent))

        DoPlanarPlot(DF, OptsL)

    LF.close()
