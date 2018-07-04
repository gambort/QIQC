import numpy as np
import matplotlib.pyplot as plt

from Libs.ReadFCHK import *
from Libs.GTOGaussian import *
from Libs.QuickElements import *
from Libs.NiceColours import *

from Libs.ConcurrenceLib import *
from Libs.ConcurrenceMisc import *

import sys
from  optparse import OptionParser

parser = OptionParser()

parser.add_option('--ListFile', type="string", default=None,
                  help="Pass a list of options via a file")

parser.add_option('--ID', type="string", default="C2H4_0",
                  help="Name of the fchk file")
parser.add_option('--Shiftz', type=float, default=0.0,
                  help="Shift perpendicular to the plane")
parser.add_option('--Range', type=float, default=-1.,
                  help="Range of the snapshot area")
parser.add_option('--ZRange', type=float, default=0.,
                  help="Range of the z axis (when applicable)")
parser.add_option('--Centre', default=False, action="store_true",
                  help="Keep the camera pointed at the trial atom")
parser.add_option('-W', type=float, default=0.0,
                  help="Amound of excited state - not recommended")

parser.add_option('--BondAxes', type="string", default=None,
                  help="BONDAXES=A-B:C-D - specifiy bonds to define the plane e.g. --BondAxes 0-1:0-2 will show the plane defined by bond vectors 0-1 and 0-2")
parser.add_option('--BondPos', type="string", default=None,
                  help="BONDPOS=K0-K1,f - specify two atom by their numbers K0 and K1 and the fraction f along their bond for the trial atom e.g. --BondPos 0-12,0.5 means the point halfway along the bond from atom 0 to atom 12")
parser.add_option('--Ghost', type="string", default=None,
                  help="GHOST=X,Y,Z - Introduce a ghost atom at point X,Y,Z to use as an additional reference e.g. --Ghost 0.,0.,0. puts a ghost at the origin")
parser.add_option('--K', type=int, default=0,
                  help="Do concurrence at atom K")

parser.add_option('--Suff', type="string", default=None,
                  help="Specify the suffix")
parser.add_option('--PlotType', type="string", default="Cs",
                  help="Choose your metric")

parser.add_option('--Show', default=False, action="store_true",
                  help="Show, but don't save the results")
parser.add_option('--SaveData', default=False, action="store_true",
                  help="Save the data - not recommended")
(Opts, args) = parser.parse_args()


Dirs = ConcFileDirs() # Initialise the various directories


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

def DoPlanarPlot(DF, Opts, Report=2):
    CsAvg, DataRange=0., [0.,250.,500.,750.,1000.]

    # Do the plots using Opts
    RAtom=DF.RAtom
    NAtom=DF.NAtom
    IDAtom=DF.IDAtom
    Basis=DF.Basis
    Orbs=DF.Orbs

    if not(Opts.Ghost is None):
        X=[float(X) for X in Opts.Ghost.split(",")]
        RAtom=np.vstack((RAtom,X))
        IDAtom
        NAtom+=1
        IDAtom+=["Gh"]

    def Permute(X,P): return np.dot(X, P)

    # Use bonds to form the plane
    RM,PU=GetPU(RAtom)
    if not(Opts.BondAxes is None) and (NAtom>2):
        B=ConcSplitOption(Opts.BondAxes)
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
        Triplets=ConcSplitOption(Opts.BondPos, Rtype="float")
        if len(Triplets)>1:
            print("# Warning! additional BondPos values will be ignored")
        K0=Triplets[0][0]
        K1=Triplets[0][1]
        f=Triplets[0][2]

    RK = (1.-f)*RAtom[K0,:] + f*RAtom[K1,:]
    SuffLoc="%d-%d,%.1f"%(K0,K1,f)

    if Opts.Centre:
        RM = RK # Centre on the bond point
    
    if Opts.Range>0.:
        Range=Opts.Range*AngToHa
    else:
        TAtom=Permute(RAtom-RM, PU)
        Range=max( np.abs(TAtom[:,0]).max(), np.abs(TAtom[:,1]).max() )*1.2
        Range=np.ceil(Range*2.+1e-4)/2.

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

    print("RM = %s"%(NiceTriplet(RM)) )
    print("RK = %s"%(NiceTriplet(RK)) )
    print("Rtl= %s"%(NiceTriplet(xyz[0,:])) )
    print("Rbr= %s"%(NiceTriplet(xyz[-1,:])) )
    print("a1 = %s"%(NiceTriplet(PU[:,0])) )
    print("a2 = %s"%(NiceTriplet(PU[:,1])) )
    print("a3 = %s"%(NiceTriplet(PU[:,2])) )

    CMID="inferno"

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
    elif Opts.PlotType=="dCs":
        dW=0.05
        CC2=Concurrence(Orbs, Basis, xyz,
                        xyz2=RK,
                        W=Opts.W)
        CC2P=Concurrence(Orbs, Basis, xyz,
                         xyz2=RK,
                         W=Opts.W+dW)

        dCs=(CC2P.Cs.reshape((Nx,Nx)) - CC2.Cs.reshape((Nx,Nx)))/dW

        Out=-1000.*dCs
        zLbl="$-1000\\partial_w C_s$"

        T=np.abs(Out).max()
        Bins=[200.,500.,1000.,2000.,5000.,1e4]
        DR=Bins[0]
        for k in range(len(Bins)-1):
            if T>Bins[k]:
                DR=Bins[k+1]
        #DR=np.ceil(np.abs(Out).max()/100.+1e-4)*100.
        if Opts.ZRange>0.:
            DR=Opts.ZRange

        DataRange=[-DR, -DR/2., 0., DR/2., DR]
        CMID="gn_inferno"
    else:
        CC2=Concurrence(Orbs, Basis, xyz,
                        xyz2=RK,
                        W=Opts.W)
        Cs=CC2.Cs.reshape((Nx,Nx))

        Out=1000.*np.maximum(Cs,0.)
        zLbl="$1000C_s$"

    print DataRange
    
    print("Maximum/minimum values: %.1f %.1f"%(Out.min(), Out.max()) )


    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    #from matplotlib.colors import LinearSegmentedColormap

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #cm_inf=LinearSegmentedColormap.from_list( \
    #    "inferno", [(0.14,0.33,0.54),
    #                (0.92,0.66,0.33),
    #                (1.00,0.95,0.75)], N=100)

    mx,my=np.meshgrid(x,x)
    surf=ax.plot_surface(mx,my,Out,
                         #cmap=cm.coolwarm,
                         cmap=NiceCMap(CMID),
                         linewidth=0, edgecolors=(0.,0.,0.,0.5),
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
    ZM=DataRange[-1]+5.
    for A in range(NAtom):
        ax.plot3D([RA[A,0],RA[A,0]],[RA[A,1],RA[A,1]],[Z0,ZM],
                  color=(0.,0.,0.), alpha=0.9, linewidth=3)
        
        AddText(ax, RA[A,0]-0.3,RA[A,1],ZM, IDAtom[A])
        
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
    ax.set_title(zLbl, fontsize=24)

    cax = fig.add_subplot(111)
    cax.set_position([0.90,0.03,0.03,0.94])
    fig.colorbar(surf, orientation='vertical',
                 ticks=DataRange,
                 extend='both',
                 cax=cax
                 )
    ax.set_position([0.03,0.03,0.94,0.94])

    Suff=""
    if not(Opts.PlotType=="Cs"):
        Suff+=Opts.PlotType
    Suff+=Opts.ID
    if not(Opts.Ghost is None):
        Suff+="Ghost"

    if Opts.Suff is None:
        Suff+="_%.1f_%.1f"%(Opts.W, Opts.Shiftz)
        Suff+="_%s"%(SuffLoc)
    else:
        Suff+="_"+Opts.Suff

    if (Report>1):
        print( "#"*10+" xyz file" )
        print( NAtom )
        print( Opts.ID )
        for k in range(NAtom):
            print( "%-3s %s"%(IDAtom[k], 
                             NiceTriplet(RAtom[k],delim=None)) )

    if Opts.SaveData:
        np.savez("%s/Plane%s"%(Dirs.Data, Suff),
                 x=x, Data=Out)

    if Opts.Show:
        plt.show()
    else:
        plt.savefig("%s/Plane%s.pdf"%(Dirs.Images_eps, Suff))
        plt.savefig("%s/Plane%s.png"%(Dirs.Images, Suff),
                    density=150)
    plt.close('all')

    return None

if Opts.ListFile is None:
    DF=DataFile("%s/%s.fchk"%(Dirs.Data, Opts.ID))
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
            print( "Loading file %s"%(IDCurrent) )
            DF=DataFile("%s/%s.fchk"%(Dirs.Data, IDCurrent))
            Report=2
        else:
            Report=1

        DoPlanarPlot(DF, OptsL, Report=Report)

    LF.close()
