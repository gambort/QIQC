import numpy as np


def fmt(Val,Kind):
    if Kind=="R":
        return float(Val)
    elif Kind=="I":
        return int(Val)
    else:
        return Val

def GetNextEntry(F):
    L=F.readline()
    if len(L)<2:
        return "EOF"

    if L[0]==" ":
        print "# Warning - seems to be a data line"
        return None,L

    Title=L[:40].rstrip()
    Kind=L[43]
    Vec=(L[47]=="N")
    Val=L[49:]

    if Vec:
        N=int(Val)
        D=[]

        if (Kind=="C"):
            for k in range(0,N,5):
                X=F.readline()
                D.extend(fmt(Y,Kind) for Y in X.split())
        else:
            while len(D)<N:
                X=F.readline()
                D.extend(fmt(Y,Kind) for Y in X.split())
        D=np.array(D)
    else:
        D=fmt(Val,Kind)

    return Title, D



class FCHKFile(dict):
    def __init__(self, FileName):
        self.FileName=FileName
        F=open(FileName, "r")
        
        self.Hdr=F.readline()
        self.Method=F.readline()

        self.AllData={}
        
        while True:
            E=GetNextEntry(F)
            if (E=="EOF"):
                break

            if E[0] is None:
                print E[1]
            self.AllData[E[0]]=E[1]
        F.close()

    def GetData(self,Tag):
        return self.AllData[Tag]

    
    def __getitem__(self,Tag):
        return self.GetData(Tag)

