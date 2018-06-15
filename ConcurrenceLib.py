import numpy as np
from Quadrature import *


HaToAng=0.52917725
AngToHa=1./HaToAng

def NiceTriplet(x, delim="("):
    if delim is None:
        return "%8.4f %8.4f %8.4f"%(x[0],x[1],x[2])
    else:
        return "(%8.4f, %8.4f, %8.4f)"%(x[0],x[1],x[2])

def GetPU(R, Show=False):
    RM=R.mean(axis=0)
    X=R-RM
    I2=np.dot(X.T,X)
    if Show: print I2

    from scipy.linalg import eigh

    w,v=eigh(I2)
    ii=np.argsort(-w)
    print NiceTriplet(w[ii])
    e1=v[:,ii[0]]
    e2=v[:,ii[1]]
    e3=np.cross(e1,e2)
    PU=np.array([ e1, e2, e3 ]).T
    return RM, PU

class Orbitals:
    def __init__(self, D):
        NShell=len(D['Shell types'])
        NPrim=len(D['Primitive exponents'])
        NBasis=D['Number of basis functions']

        print "N_Shell = %4d, N_Prim = %4d, N_Basis = %4d"\
            %(NShell, NPrim, NBasis)

        Nalpha=D['Number of alpha electrons']
        Nbeta=D['Number of beta electrons']
        
        if (Nalpha-Nbeta)==0:
            NOrbitals=D['Alpha MO coefficients'].shape[0]\
                /D['Number of basis functions']
            phiCoeffs=D['Alpha MO coefficients'].reshape(NOrbitals,NBasis)

            Ens=D['Alpha Orbital Energies']

            f=np.zeros((NOrbitals,))
            f[:Nalpha]=1.

            NH=Nalpha-1
            EH=Ens[NH]
            NH2=NH
            for kp in range(max(0,NH-12),NH):
                if np.abs(Ens[kp]-EH)<1e-4:
                    NH2-=1

            NL=Nalpha
            EL=Ens[NL]
            NL2=NL
            for kp in range(NL+1,min(NL+12,NOrbitals)):
                if np.abs(Ens[kp]-EL)<1e-6:
                    NL2+=1

            if not((NL2-NL)==0) or not((NH2-NH)==0):
                print "%d-fold degenerate HOMO, %d-fold degenerate LUMO"\
                    %(NH-NH2+1, NL2-NL+1)
                print "Frontier from %d to %d"%(NH2, NL2),\
                    Ens[NH2:(NL2+1)]

        else:
            print "Spin unpolarized systems are not implemented"

        self.f=f
        self.NBasis=NBasis
        self.NH=[NH2,NH]
        self.NL=[NL,NL2]
        self.phiCoeffs=phiCoeffs

""" Helper function """
def GetDM(phi, phib, S=1.0, DiagOnly=False):
    if DiagOnly:
        X=S*phi**2
        return X, X, X
    else:
        return S*phi**2, S*phib**2, S*np.outer(phi,phib)

""" The following reads the density, HOMO and LUMO orbitals
onto the rays between different atoms """
class Concurrence:
    def __init__(self, Orbs, Basis, xyz,
                 xyz2=None, DiagOnly=False,
                 W=0.3,
                 RPrune=5., cPrune=1e-4):
        f=Orbs.f
        NL,NH=Orbs.NL,Orbs.NH

        DH=1./(NH[1]-NH[0]+1.) # Scale factor for HOMOs
        DL=1./(NL[1]-NL[0]+1.) # Scale factor for LUMOs

        phiCoeffs=Orbs.phiCoeffs

        nG=0. # total density n
        nG2=0.
        rhoG=0.
        
        nH=0.
        nH2=0.
        rhoH=0.

        nL=0.
        nL2=0.
        rhoL=0.

        
        if not(xyz2 is None) and (len(xyz2.shape)==1):
            xyz2.resize((1,3))

        for b in range(NL[1]+1):
            phib=Basis.GetOrbGrid(phiCoeffs[b,:], xyz,
                                  RPrune=RPrune, cPrune=cPrune)
            if not(xyz2 is None):
                phib2=Basis.GetOrbGrid(phiCoeffs[b,:], xyz2,
                                       RPrune=RPrune, cPrune=cPrune)
            else:
                phib2=phib

            nb,nb2,rhob=GetDM(phib, phib2, S=1.0, DiagOnly=DiagOnly)
            if f[b]>0.:
                nG+=f[b]*nb
                nG2+=f[b]*nb2
                rhoG+=f[b]*rhob

            if (b>=NH[0]) and (b<=NH[1]):
                nH+=DH*nb
                nH2+=DH*nb2
                rhoH+=DH*rhob
            if (b>=NL[0]) and (b<=NL[1]):
                nL+=DL*nb
                nL2+=DL*nb2
                rhoL+=DL*rhob

        nU=nG+nL
        nU2=nG2+nL2
        rhoU=rhoG+rhoL

        nD=nG-nH
        nD2=nG2-nH2
        rhoD=rhoG-rhoH

        if DiagOnly:
            n2S=nG*nG2-rhoG*rhoG
            n2D=nG*nG2
        else:
            n2S=np.outer(nG,nG2)-rhoG*rhoG
            n2D=np.outer(nG,nG2)

        n2G=2.*(n2S+n2D)
        
        if DiagOnly:
            n2S2=nU*nU2 - rhoU*rhoU \
                + nD*nD2 - rhoD*rhoD
            n2D2=nU*nD2 + nD*nU2
        else:
            n2S2=np.outer(nU,nU2) - rhoU*rhoU \
                + np.outer(nD,nD2) - rhoD*rhoD
            n2D2=np.outer(nU,nD2) + np.outer(nD,nU2)
        
        n2E=n2S2 + n2D2

        self.nG=nG
        self.nE=nU+nD

        self.nG2=nG2
        self.nE2=nU2+nD2

        self.nL=nL
        self.nH=nH

        self.n2G=n2G
        self.n2E=n2E

        self.n=2.*(1.-W)*self.nG + W*self.nE
        self.n2=2.*(1.-W)*self.n2G + W*self.n2E

        # Calculate the concurrence
        self.p=1.-4./3.*(W*n2E)/( (1.-W)*n2G + W*n2E )
        self.Cs2El=np.maximum( (3.*self.p-1.)/2., 0 )

        nD=nL-nH
        nD2=nL2-nH2
        rhoD=rhoL-rhoH

        if DiagOnly:
            X0=nG*nG2
            X0D=nG*nD2 + nD*nG2
            Xhl=nH*nL2 + nL*nH2
        else:
            X0=np.outer(nG,nG2)
            X0D=np.outer(nG,nD2) + np.outer(nD,nG2)
            Xhl=np.outer(nH,nL2) + np.outer(nL,nH2)


        Y0=rhoG*rhoG
        Y0D=rhoG*rhoD
        Yhl=rhoH*rhoL

        A=(X0-Y0) + W/2.*(X0D-2.*Y0D) - W/6.*(Xhl-2.*Yhl)
        B=X0 + W/2.*X0D - W/3.*(Xhl+Yhl)
        F=Y0 + W*Y0D - W/6.*(Xhl+4.*Yhl)

        #print (F-A).max()
        #print (-B).max()

        #Cs=(4.*Y0-2.*X0 + W*(4.*Y0D - X0D) - 2.*W*Yhl) \
        #    / (4.*X0-2.*Y0 + 2.*W*(X0D - Y0D) - W*Xhl)
        self.Cs=np.maximum(-B/(A+B), (F-A)/(A+B))
        #self.Cs=np.maximum( 0., (F-A)/(A+B) )

        N=4.*Y0-2.*X0+W*(4.*Y0D-X0D)+W*(3.*Xhl-2.*Yhl)
        D=4.*X0-2.*Y0+2.*W*(X0D-Y0D)+W*(4.*Yhl-Xhl)
        self.CsSinglet=np.maximum(0., N/D)

        self.rs=0.6203505/self.nG**2


