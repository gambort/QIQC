import numpy as np
from Elements import ELEMENTS

HaToAng=0.52917725
AngToHa=1./HaToAng

def GetBonds(Pos, IDs, Units="Ang"):
    N=len(IDs)
    if Pos.shape[0]!=N:
        print("# Mismatch in size")
        return None, None

    R=Pos
    if Units=="Ha":
        R=Pos*HaToAng
    if Units=="pm":
        R=Pos/100.

    # Get atomic radii in angstrom
    RC=np.zeros( (N,) )
    RA=np.zeros( (N,) )
    for I in range(N):
        RC[I]=ELEMENTS[IDs[I]].covrad
        RA[I]=ELEMENTS[IDs[I]].atmrad
    RB=RC # np.maximum(RA, RC)

    Bonds=np.zeros( (N,N) )
    BondList=[]
    for I in range(N):
        for J in range(I+1,N):
            XIJ=R[I,:]-R[J,:]
            RIJ=np.sqrt( (XIJ**2).sum() )
            RCIJ=(RB[I]+RB[J])*1.2
            
            if RIJ<RCIJ:
                Bonds[I,J]=1.
                Bonds[J,I]=1.
                BondList+=[(I,J)]
    return Bonds, BondList
                
