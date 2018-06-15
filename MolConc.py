import numpy as np

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


from  optparse import OptionParser

parser = OptionParser()
parser.add_option('--ID', type="string", default="HCl-127")
parser.add_option('--Grid', type="string", default="CO")
parser.add_option('-W', type=float, default=0.1)

parser.add_option('--PlotType', type=int, default=1)
parser.add_option('--WriteData', type=int, default=0)

(Opts, args) = parser.parse_args()

Pre=Opts.ID.split(".")[0]
W=Opts.W


fs=16
fss=12

class MoleculeData:
    def __init__(self, Pre, Grid, Type="All"):
        self.z=Grid['z']
        self.r=Grid['r']
        self.theta=Grid['theta']
        self.wtheta=Grid['wtheta']
        self.Nz=len(self.z)
        self.Nr=len(self.r)
        self.Ntheta=len(self.theta)

        RhoData=np.loadtxt("Data/"+Pre+".ray", skiprows=8)

        NTot=self.Nz*self.Nr*self.Ntheta
        
        if not(RhoData.shape[0]==NTot):
            print("Grid is inconsistent with data in %s.ray")
            quit()

        self.Occ=np.loadtxt("Data/"+Pre+"-OccA.ray", skiprows=9)[:,3:]
        self.LUMO=np.loadtxt("Data/"+Pre+"-LUMO.ray",
                             skiprows=9, usecols=(3,))\
                             .reshape( (NTot,-1,) )

        self.xyz=RhoData[:,0:3]
        self.Rho=RhoData[:,3].reshape( (NTot,-1,) )

        #self.Average(self.Rho)
        #self.Average(self.Occ)
        #self.Average(self.LUMO)

        self.NOcc=self.Occ.shape[1]

        self.SetMask(Type)

    def Average(self,A):
        T=1.*A
        NLine=self.Nz*self.Nr
        A=np.zeros((NLine,T.shape[1]))
        for k in range(self.Ntheta):
            i0=k*NLine
            A+=self.wtheta[k]*T[i0:(i0+NLine),:]

    def SetMask(self, Type="All"):
        if Type=="All":
            # Every 2nd element to save memory
            self.Mask=np.array(range(self.Rho.shape[0]), dtype="int")
        else:
            # Line along the z axis
            self.Mask=np.array(range(self.Nz), dtype="int")
            
    def GetDens(self):
        return self.Rho

    def GetOrb(self, j):
        if j==self.NOcc:
            return self.LUMO[self.Mask,0]
        elif j>self.NOcc:
            return False
        else:
            return self.Occ[self.Mask,j]

    def GetnRho(self, f=None, DiagOnly=False):
        if f is None:
            f=np.ones(self.NOcc)
        elif f is "+":
            f=np.ones(self.NOcc+1)
        elif f is "-":
            f=np.ones(self.NOcc-1)

        n,rho=0.,0.
        for k in range(len(f)):
            phik=self.GetOrb(k)
            n+=f[k]*phik**2
            if not(DiagOnly):
                rho+=f[k]*np.outer(phik,phik)
        if DiagOnly: rho=n

        return n,rho

    def Calcn2(self, W=0., Flag=True, DiagOnly=False):
        Nz=self.Nz

        nG,rhoG=self.GetnRho(DiagOnly=DiagOnly)

        nU,rhoU=self.GetnRho("+",DiagOnly=DiagOnly)
        nD,rhoD=self.GetnRho("-",DiagOnly=DiagOnly)
    
        if not(DiagOnly):
            n2S=np.outer(nG,nG)-rhoG*rhoG
            n2D=np.outer(nG,nG)

            n2G=2.*(n2S+n2D)

            n2S2=np.outer(nU,nU)-rhoU*rhoU + np.outer(nD,nD)-rhoD*rhoD
            n2D2=np.outer(nU,nD) + np.outer(nD,nU)

            n2E=n2S2 + n2D2
        else:
            n2S=0.
            n2D=nG**2

            n2G=2*n2D

            n2S2=0.
            n2D2=2.*nU*nD

            n2E=n2D2

        self.nG=2.*nG
        self.nE=nU+nD

        self.n2G=n2G
        self.n2E=n2E


        self.n=(1.-W)*self.nG + W*self.nE
        self.n2=(1.-W)*self.n2G + W*self.n2E

        if not(DiagOnly):
            self.G2=self.n2/np.outer(self.n,self.n)
        else:
            self.G2=self.n2/self.n**2
    
        # Calculate the concurrence
        p=1.-4./3.*(W*n2E)/( (1.-W)*n2G + W*n2E )
        self.Cs=np.maximum( (3.*p-1.)/2., 0 )

        return self.n2
        

Grid=np.load("Data/%s.grid.npz"%(Opts.Grid))
Molecule=MoleculeData(Pre, Grid)

if Opts.PlotType==0:
    fig = plt.figure()
    fig.set_size_inches(8.,8.,forward=True)
    
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(30,30)
    ax.set_position([0.0,0.03,1.0,0.97])
    #ax.set_clim(0.,1.)

    Molecule.SetMask(Type="Line")
    Molecule.Calcn2(W=W)

    
    z=Grid['z']
    Z,Zp=np.meshgrid(z,z)
    ax.plot_surface(Z,Zp,Molecule.Cs,rstride=1,cstride=1,
                    cmap=cm.RdYlBu,linewidth=0,
                    vmin=0., vmax=1.)


    ax.set_zlim3d([0.,1.])
    ax.set_xlabel("$z'$", fontsize=fs)
    ax.set_ylabel("$z$", fontsize=fs)
    ax.text2D(0.05,0.8, "$C_s$", fontsize=fs,
              transform=ax.transAxes)    

    plt.savefig("Images/MolConc_%s_%.2f.png"%(Pre,W), dpi=150)


if Opts.PlotType==1:
    fig = plt.figure()
    fig.set_size_inches(6.,3.,forward=True)
    
    ax = fig.add_subplot(111)
    ax.set_position([0.1,0.18,0.88,0.8])

    Molecule.Calcn2(W=W, DiagOnly=True)

    z=Grid['z']

    rho=Molecule.nG
    rs=0.62035*rho**(-0.3333333333)
    Cd=Molecule.Cs
    #Cd=np.diag(Moleculue.Cs)

    ax.scatter(np.sqrt(rs),Cd,
               alpha=0.8, edgecolors=None)

    ax.set_xlim([0.,8.])
    ax.set_ylim([-0.05,1.05])

    ax.set_xlabel("$r_s^{0.5}$", fontsize=fss)
    ax.set_ylabel("$C_s(r,r)$", fontsize=fss)

    #plt.show()
    plt.savefig("Images/MolDensConc_%s_%.2f.png"%(Pre,W), dpi=150)

if Opts.PlotType==100:
    import matplotlib.pyplot as plt

    for j in range(4,Molecule.NOcc+1):
        plt.plot(Grid['z'], Molecule.GetOrb(j)[:Molecule.Nz], linewidth=4)
    plt.show()
