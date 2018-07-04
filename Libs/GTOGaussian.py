import numpy as np

# Order
# Plus
# S,    X,Y,Z,    XX,YY,ZZ,XY,XZ,YZ,
# XXX,YYY,ZZZ,XYY,XXY,XXZ,XZZ,YZZ,YYZ,XYZ
# Minus
# S,    X,Y,Z,    3ZZ-RR,XZ,YZ,XX-YY,XY,
# ZZZ-ZRR,XZZ-XRR,YZZ-YRR,XXZ-YYZ,XYZ,XXX-XYY,XXY-YYY,
# XY(XX-YY)

NlConst=[2.526475110984259, 2.917322170855304, 2.609332274519885,
         1.972469796089753, 1.314979864059836]
NInShell={0:1,1:3,-1:4,2:6,-2:5,3:10,-3:7,-4:9}


# Get the normalized polynomial term
def GetNormPoly(L,m=0,xyz=None):
    N=1.
    if L==1: N=np.sqrt(3.)
    if L==-2:
        N=np.sqrt([ 5./4.,15.,15.,15./4.,15. ][m])
    if L==-3:
        N=np.sqrt([ 105./8., 35./8., 35./8.,
                    105./4., 105., 35./4., 35./4. ][m])

    return N*GetPoly(L,m,xyz)/np.sqrt(4.*np.pi)
# Get the polynomial term
def GetPoly(L,m=0,xyz=None):
    if L==0:
        return 1.
    if L==1:
        return xyz[:,m]
    if L==2:
        T=[(2,0,0),(0,2,0),(0,0,2),(1,1,0),(1,0,1),(0,1,1)][m]
        return xyz[:,0]**T[0]*xyz[:,1]**T[1]*xyz[:,2]**T[2]
    if L==-2:
        if m==0:
            return 2.*xyz[:,2]**2-xyz[:,0]**2-xyz[:,1]**2
        elif m==1:
            return xyz[:,0]*xyz[:,2]
        elif m==2:
            return xyz[:,1]*xyz[:,2]
        elif m==3:
            return xyz[:,0]**2-xyz[:,1]**2
        elif m==4:
            return xyz[:,0]*xyz[:,1]
    if L==3:
# XXX,YYY,ZZZ,XYY,XXY,XXZ,XZZ,YZZ,YYZ,XYZ
        T=[(3,0,0),(0,3,0),(0,0,3),
           (1,2,0),(2,1,0),(2,0,1),
           (1,0,2),(0,1,2),(0,2,1),
           (1,1,1)][m]
        return xyz[:,0]**T[0]*xyz[:,1]**T[1]*xyz[:,2]**T[2]
    if L==-3:
        if m==0:
            return -xyz[:,2]*(xyz[:,0]**2+xyz[:,1]**2)
        if m==1:
            return -xyz[:,0]*(xyz[:,0]**2+xyz[:,1]**2)
        if m==2:
            return -xyz[:,1]*(xyz[:,0]**2+xyz[:,1]**2)
        if m==3:
            return xyz[:,2]*(xyz[:,0]**2-xyz[:,1]**2)
        if m==4:
            return xyz[:,0]*xyz[:,1]*xyz[:,2]
        if m==5:
            return xyz[:,0]*(xyz[:,0]**2-xyz[:,1]**2)
        if m==6:
            return xyz[:,1]*(xyz[:,0]**2-xyz[:,1]**2)
    if L==-4:
        print("# L=-4 Not implemented")
        return 0.

    return None

class GTOGaussian:
    def __init__(self, FCHKData):
        self.FCHKData=FCHKData

        self.InitialiseBasis()

    def InitialiseBasis(self):
        D=self.FCHKData # Short hand for FCHK Data

        self.NShell=len(D['Shell types'])
        self.NPrim=len(D['Primitive exponents'])
        self.NBasis=D['Number of basis functions']

        k0=0
        GB=[] # Initialise the basis
        CC=D['Coordinates of each shell'].reshape(self.NShell,3)

        for k in range(self.NShell):
            GB.append({})
            NS=D['Number of primitives per shell'][k]
            X=D['Contraction coefficients'][k0:(k0+NS)]

            GB[k]['Coeff']=X
            GB[k]['Indx']=np.arange(k0,k0+NS)
            GB[k]['alpha']=D['Primitive exponents'][k0:(k0+NS)]
            GB[k]['l']=D['Shell types'][k]
            GB[k]['Position']=CC[k,:]

            l=int(np.abs(GB[k]['l']))
            GB[k]['Norm']=GB[k]['alpha']**((1.5+l)/2.)*NlConst[l]

            k0+=NS
    
        self.GaussBasis=GB

        self.FullBasis=[]
        NT=0
        for k in range(self.NShell):
            l=GB[k]['l']
            NS=NInShell[l]
            for kp in range(NT,NT+NS):
                m=kp-NT
                self.FullBasis.append((k,l,m))
            NT+=NS

    def GetGaussianProps(self, b):
        k,l,m=self.FullBasis[b]
        gb=self.GaussBasis[k]
        return gb, l, m

    def GetGaussian(self, b, xyz, RPrune=6.0):
        k,l,m=self.FullBasis[b]
        gb=self.GaussBasis[k]

        xyzS=xyz-gb['Position']
        RS2=xyzS[:,0]**2+xyzS[:,1]**2+xyzS[:,2]**2

        DMin2=np.min(RS2)*np.min(gb['alpha']) # minimum distance 
        if DMin2>RPrune**2:
            return 0.

        P=GetNormPoly( l, m, xyzS )

        G=0.
        for s in range(len(gb['Coeff'])):
            G+=gb['Norm'][s]*gb['Coeff'][s] \
                * np.exp(-gb['alpha'][s]*RS2)
        
        return P*G

    def GetOrbGrid(self, Orb, xyz, RPrune=6.0, cPrune=1e-5):
        GG=0.
        if len(xyz.shape)==1:
            xyz=xyz.reshape((1,3))

        for b in range(self.NBasis):
            cc=Orb[b]
            if np.abs(cc)<=cPrune:
                continue
            gb,l,m=self.GetGaussianProps(b)
            if (np.abs(l)>=4):
                if np.abs(cc)>1e-2:
                    print("# Warning: >g orbitals not implemented: "\
                              +"missing weight is %.3f"%(cc))
            else:
                BB=self.GetGaussian(b, xyz, RPrune=RPrune)
                GG+=cc*BB
        return GG
