###############################################################
#
# Implements concurrence and related metrics in Python
#
# Reads a molecule from a .fchk file (e.g. from Gaussian) and
# calculates properties using a condensed metric
#
# Outputs are a viable yaml file
#
# Run ConcPairs.py --help for help
#
###############################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh


from Libs.ReadFCHK import *
from Libs.GTOGaussian import *

from Libs.QuickElements import *
from Libs.NiceColours import *

from Libs.SphGrid import *
from Libs.GetBonds import *

from Libs.ConcurrenceLib import *
from Libs.ConcurrenceMisc import *

from  optparse import OptionParser

parser = OptionParser()
parser.add_option('--ID', type="string", default="HCl-127",
                  help="Name of the fchk file")


parser.add_option('--Pairs', type="string", default=None,
                  help="K0-K1,N01:K1-K2,N12 - List of atom pairs and number of steps between them, separated by : e.g. 1-0:4,0:12,4 steps between atoms 1 and 0 and 0 and 12 divided into quarters")
parser.add_option('--Ghost', type="string", default=None,
                  help="GHOST=X,Y,Z - Introduce a ghost atom at point X,Y,Z to use as an additional reference e.g. --Ghost 0.,0.,0. puts a ghost at the origin")

parser.add_option('-W', type=float, default=0.0,
                  help="Amount of excited state - not recommended")

parser.add_option('--NGrid', type=int, default=11,
                  help="Size of each dimension of the Gaussian grid")
parser.add_option('--Radius', type=float, default=-1.,
                  help="Radius of Gaussian envelope")

parser.add_option('--Mode', type="string", default="rsCs",
                  help="Type of metric")

parser.add_option('--Tensor', default=False, action="store_true",
                  help="Show tensor metric as well")

parser.add_option('--Surface', default=True, action="store_true",
                  help="Surface of a sphere")

parser.add_option('--Show', default=False, action="store_true",
                  help="Show the concurrence in a plot")

parser.add_option('--Proj', type="string", default=None)
parser.add_option('--PHeight', type="float", default=None)

parser.add_option('--Step', type=float, default=0.25)
parser.add_option('--Shiftx', type=float, default=0.)
parser.add_option('--Shifty', type=float, default=0.)
parser.add_option('--Shiftz', type=float, default=0.)

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

Dirs = ConcFileDirs() # Initialise the various directories

L=Logger("%s/Pairs-%s-%.1f.log"%(Dirs.Data, Opts.ID, Opts.Radius))

RayShift=np.array([Opts.Shiftx,Opts.Shifty,Opts.Shiftz])*AngToHa

#######################################################
D=FCHKFile("%s/%s.fchk"%(Dirs.Data, Opts.ID))
ZAtom=D['Nuclear charges']
IDAtom=[QE.GetID(Z) for Z in ZAtom]

NAtom=len(ZAtom)
RAtom=D['Current cartesian coordinates'].reshape(NAtom,3)

Bonds,BondList=GetBonds(RAtom, IDAtom, Units="Ha")

if not(Opts.Ghost is None):
    X=[float(X)*AngToHa for X in Opts.Ghost.split(",")]
    RAtom=np.vstack((RAtom,X))
    NAtom+=1
    ZAtom=np.hstack((ZAtom,[0]))
    IDAtom+=["Gh",]
    BB=np.zeros((NAtom,NAtom))
    BB[:(NAtom-1),:(NAtom-1)]=Bonds
    Bonds=BB

#######################################################
RMin2=1e10;
for A in range(NAtom):
    for Ap in range(A+1,NAtom):
        R2=((RAtom[A,:]-RAtom[Ap,:])**2).sum()
        RMin2=min(R2,RMin2)

L.Log("Atoms:")
for A in range(NAtom):
    #L.Log("# %-3s %3d %s [Ang]"%(IDAtom[A], ZAtom[A],
    #                             NiceTriplet(RAtom[A,:]*HaToAng)))
    AtID=IDAtom[A]+"%d"%(A)
    L.Log("  %-6s: { ID : %2s, Z: %3d, R: %s, Units: Ang }"\
              %(AtID, IDAtom[A], ZAtom[A], NiceTriplet(RAtom[A,:]*HaToAng)))

BStr=",".join( "[ %d, %d ]"%(B[0],B[1]) for B in BondList )
L.Log("Bonds: [ "+ BStr + " ]")

#######################################################
# Some default pairs
AtomPairs=[(0,1),]
#######################################################
# Process the bond positions into pairs
if not(Opts.Pairs is None):
    AtomPairs=ConcSplitOption(Opts.Pairs, Rtype="int")
elif len(BondList)>1:
    def GoTo(BondList, Curr, Visited):
        N=-1
        for B in BondList:
            if Curr==B[0]: N=B[1]
            if Curr==B[1]: N=B[0]
            if (N>=0) and not(N in Visited):
                return N
        return None
    Curr=0
    Visited=[Curr,]
    N=GoTo(BondList, Curr, Visited)
    if not(N is None):
        AtomPairs=[(Curr,N)]
    while not(N is None):
        Visited+=[N,]
        Curr=N
        N=GoTo(BondList, Curr, Visited)
        if not(N is None):
            AtomPairs+=[(Curr,N)]




#######################################################
# Setup basis and orbitals
Basis=GTOGaussian(D)
Orbs=Orbitals(D)


#######################################################
# Use shortest bond to specify radius if set negative
if Opts.Radius>0.:
    Radius=Opts.Radius*AngToHa
else:
    Radius=1.

L.Log("# Shortest bond:  %.4f [Ang]"%(np.sqrt(RMin2)*HaToAng))
L.Log("# Gausssian rad.: %.4f [Ang]"%(Radius*HaToAng))

#######################################################
NGrid=Opts.NGrid
NGrid3=NGrid**3

L.Log("Grid info: >")
if Opts.Surface:
    Cube, W = GetSphGrid(Radius)
    if Opts.Radius>0.:
        L.Log("  Using surface grid at Radius = %.3f"%(Radius))
    else:
        L.Log("  Using surface grid at Radius = %.2f rs"%(-Opts.Radius))
else:
    x,wx,lwx=GaussHermiteWeights(NGrid,S=Radius)
    o=0.*x+1.

    L.Log("  Using gaussian grid %d^3 with Radius = %.2f ;"%(NGrid, Radius))
    L.Log("  Range from %.3f to %.3f"%(x[0], x[-1]))

    def outrio(a,b,c):
        X=np.outer(a,b)
        return np.einsum(X,[0,1],c,[2,],[0,1,2])

    X=outrio(x,o,o).reshape((NGrid3,))
    Y=outrio(o,x,o).reshape((NGrid3,))
    Z=outrio(o,o,x).reshape((NGrid3,))
    R2=X**2+Y**2+Z**2

    W=outrio(wx,wx,wx).reshape([NGrid3,])
    W*=np.exp(-R2/Radius**2)
    W/=np.sum(W)

    Cube=np.zeros((NGrid3,3))
    Cube[:,0]=X
    Cube[:,1]=Y
    Cube[:,2]=Z

#######################################################

def GetCsAvg(Cs, W, Cube=None):
    CsM=np.dot(Cs, W)
    if Cube is None:
        return CsM

    CsM=max(1e-20,CsM)
    v=np.zeros((3,))
    for i in range(0,3):
        v[i]=np.dot(Cs*Cube[:,i],W)

    T=np.zeros((3,3))
    for i in range(0,3):
        for j in range(i,3):
            T[i,j]=np.dot(Cs*Cube[:,i]*Cube[:,j], W)
            if not(i==j): T[j,i]=T[i,j]
    return CsM, v, T

def NeatVec(T):
    return "[ "+", ".join(["%6.3f"%(X) for X in T.tolist()])+" ]";
def NeatTens(T):
    Rows=[]
    for i in range(3):
        Rows+=[NeatVec(T[:,i])]
    return "[ "+", ".join(Rows)+" ]";

#######################################################

L.Log("Pairs:")
A2Old=-5
Index=0
AllData=[]
for AP in AtomPairs:
    if (len(AP)==2):
        A,A2=AP
        N=0
    else:
        A,A2,N=AP

    if A==A2: continue

    RA=RAtom[A,:]
    RA2=RAtom[A2,:]

    if (N==0):
        N=int(np.ceil( np.sqrt(((RA2-RA)**2).sum())/Opts.Step ))

    AtID=IDAtom[A]+"%d"%(A)
    AtID2=IDAtom[A2]+"%d"%(A2)

    L.Log("# From: %-6s to %-6s"%(AtID, AtID2))
    for n in range(N+1):
        f=float(n)/float(N)
        RP=RA*(1.-f) + RA2*f

        if (A==A2Old) and (f==0.):
            continue

        R0=RP+RayShift


        xyz=R0.reshape((1,3))
        CC=Concurrence(Orbs, Basis, xyz, W=Opts.W)
        rs=CC.rs[0]

        if Opts.Radius<0.:
            Radius=-Opts.Radius*rs
            xyz2=R0+Radius*Cube
        else:
            xyz2=R0+Cube

        if Opts.Mode=="None":
            Cs=0.*xyz2[:,0]
        elif Opts.Mode=="Cs2":
            CC=Concurrence(Orbs, Basis, xyz, xyz2=xyz2, W=Opts.W)
            Cs=CC.Cs2El
        elif Opts.Mode=="CsS":
            CC=Concurrence(Orbs, Basis, xyz, xyz2=xyz2, W=Opts.W)
            Cs=CC.CsSinglet
        elif Opts.Mode=="dCs" or Opts.Mode=="rsdCs":
            CC=Concurrence(Orbs, Basis, xyz, xyz2=xyz2, W=Opts.W+1e-7)
            Cs=CC.Cs
            CsA1=GetCsAvg(Cs, W)

            dW=0.05
            CC2=Concurrence(Orbs, Basis, xyz, xyz2=xyz2, W=Opts.W+dW)
            Cs2=CC2.Cs
            CsA2=GetCsAvg(Cs2, W)

            CsAM=(CsA2+CsA1)/2.+1e-10

            LCsA=(CsA2-CsA1)/dW / CsAM
            
            Cs=0.5+(Cs2-Cs)/dW
        else:
            CC=Concurrence(Orbs, Basis, xyz, xyz2=xyz2, W=Opts.W)
            Cs=CC.Cs


        Surf=4.*np.pi*Radius**2
        if Opts.Tensor:
            CsA,CsAv,CsAT=GetCsAvg(Cs, W, Cube)

            CsAvN=CsAv/CsA
            CsATN=CsAT/CsA

            EVal,EVec=eigh(CsATN)
        else:
            CsA=GetCsAvg(Cs, W)

        CsA=float(CsA)
        if Opts.Mode=="dCs" or Opts.Mode=="rsdCs":
            CsA=LCsA

        Data={ 'Step':Index, 'Cs': CsA, 'R':RP*HaToAng, 'rs':rs,
               'f':f, 'df':1./float(N),
               'K1':A, 'K2':A2,
               'A1':AtID, 'A2':AtID2,
               'R1':RA*HaToAng, 'R2':RA2*HaToAng }

        Hdr="Step : %3d, Cs : %5.0f, R : %s, rs : %.3f, "\
            %(Index, 1000.*CsA, NeatVec(RP*HaToAng), rs)
        Hdr+="f : %.3f, A1: %-6s, A2: %-6s"%(f, AtID, AtID2)
        if Opts.Tensor:
            L.Log("- { %s, "%(Hdr) )
            L.Log("    Csv:  "+NeatVec(CsAvN)+",")
            L.Log("    CsT:  "+NeatTens(CsATN)+",")
            L.Log("    EVal: "+NeatVec(EVal)+",")
            L.Log("    EVec: "+NeatTens(EVec)+",")
            L.Log("}")

            Data['Csv']=CsAvN
            Data['CsT']=CsATN
        else:
            L.Log("- { %s }"%(Hdr) )


        AllData+=[Data]

        Index+=1 # Step up the index
    A2Old=A2 # Use to skip dupes

# Tidy the log
L.Close()

# Will be given an option
if Opts.Show:
    x0=0.
    A1=AllData[0]['A1']
    A2=AllData[0]['A2']
    x=np.zeros((len(AllData),))
    y=np.zeros((len(AllData),))
    yp=np.zeros((len(AllData),))

    Lbls=[(A1,x0),]
    for I in range(len(AllData)):
        D=AllData[I]
        dx=np.sqrt( ((D['R2']-D['R1'])**2).sum() )*D['df']
        x[I]=x0
        y[I]=D['Cs']
        yp[I]=D['rs']*HaToAng

        if not(D['A1']==A1):
            A1=D['A1']
            if A2==A1:
                Lbls+=[(A1,x[I-1])]
            else:
                Lbls+=[(A2+"/"+A1,x[I-1])]
        A2=D['A2']
        x0+=dx
    Lbls+=[(A2,x[-1])]

    xt=[X[1] for X in Lbls]
    xl=[X[0] for X in Lbls]

    ax=plt.gca()
    ax.plot(x, y, "-", linewidth=4,
            label="$C_s$")
    ax.plot(x, yp, ":", linewidth=4,
            label="$r_s$ [$\\AA$]")
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, fontsize=16)
    yt=[0.,0.25,0.50,0.75,1.00]
    ax.set_yticks(yt)
    ax.set_yticklabels(yt, fontsize=16)
    #ax.axis([0,xt[-1],-0.03,1.03])
    plt.legend(loc="upper center", fontsize=18)
    #plt.tight_layout()


    if Opts.Mode=="Cs":
        Suff="_%s_%.1f"%(Opts.ID, Opts.Radius)
    else:
        Suff="%s_%s_%.1f"%(Opts.Mode, Opts.ID, Opts.Radius)

    plt.savefig("%s/Pairs%s.eps"%(Dirs.Images, Suff))

    plt.clf()

    from Libs.RenderMol import *
    
    ax=plt.gca()
    
    NData=len(AllData)
    ListData=np.zeros((NData,2))
    D=AllData[0]
    K1,K2=D['K1'],D['K2']
    Proj=[0,K1,K2]
    PVI={}
    Index0=0
    for I in range(NData):
        D=AllData[I]
        ListData[I,0]=D['Cs']
        ListData[I,1]=D['rs']
        if not(D['K1']==K1 and D['K2']==K2) or I==(NData-1):
            if (I==(NData-1)):
                IndexF=NData
            else:
                IndexF=I
            PVI[(K1,K2)]=(Index0,IndexF)
            Proj+=[K1,K2]

            Index0=I
            if D['K1']==K2:
                Index0-=1 # connector

            K1=D['K1']
            K2=D['K2']

    PV={}
    for Pair in PVI:
        I0,I1=PVI[Pair]
        N=(I1-I0)
        y =ListData[I0:I1,0].reshape((N,))
        yp=ListData[I0:I1,1].reshape((N,))
        x=np.linspace(0.,1.,N)
        PV[Pair]=(x,y,yp)

    Proj=list(set(Proj))[:3]

    if not(Opts.Proj is None):
        R=ConcSplitOption(Opts.Proj, Rtype="int")
        Proj=[ R[0][0],R[0][1],R[1][0],R[1][1] ]

    PCols=("r-", "g--")
    if not(Opts.Mode=="Cs"):
        PCols=("b-", "g--")
    if Opts.Mode=="rsCs" or Opts.Mode=="rsdCs":
        # Special mode that shows "Cs" and "rs Cs" (or dCs)
        for T in PV:
            x,y,yp=PV[T]
            yp=y*yp
            PV[T]=(x,y,yp)
        PCols=("r-", "b--")
    RenderMol(ax, ZAtom, RAtom, Proj=Proj,
              Bonds=Bonds, PlotValues=PV,
              PHeight=Opts.PHeight,
              PCols=PCols)

    plt.tight_layout()

    plt.savefig("%s/MolPairs%s.png"%(Dirs.Images, Suff))
    plt.savefig("%s/MolPairs%s.pdf"%(Dirs.Images_eps, Suff))
    #plt.show()
