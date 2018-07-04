#### Helper functions for Concurrence stuff
import os

class ConcFileDirs:
    def __init__(self,
                 Images="Images",
                 Data="Data",
                 eps="eps" ):

        if os.path.isfile("ConcDirs.cfg"):
            import yaml
            F=open("ConcDirs.cfg", "r")
            CFG=yaml.load(F)
            F.close()
            if "Images" in CFG: Images=CFG["Images"]
            if "Data" in CFG: Data=CFG["Data"]
            if "eps" in CFG: eps=CFG["eps"]
        
        self.Images = Images
        self.Data = Data
        if eps is None:
            self.Images_eps = self.Images
        else:
            self.Images_eps = os.path.join(self.Images, eps)

        self.CreateAll()


    def CreateAll(self):
        for D in self.AllDirs():
            try:
                os.makedirs(D)
            except OSError:
                if not os.path.isdir(D):
                    raise
        
    def AllDirs(self): # Return all directories
        return [self.Images, self.Data, self.Images_eps]
        

def ConcSplitOption(Option, Rtype="int" ):
    if Option is None:
        return None

    if Rtype=="int":
        fOpt=lambda x: int(x)
    else:
        fOpt=lambda x: float(x)
    
    List=Option.split(":") # the delimiter for units
    NOpt=len(List)
    Vals=[None,]*NOpt

    for k in range(NOpt):
        Opt=List[k]
        
        T=Opt.split("-")
        if len(T)<2:
            Vals[k]=None
            continue
        
        P1=int(T[0])
        T=T[1].split(",")
        if len(T)<2:
            P2=int(T[0])
            X=None
        else:
            P2=int(T[0])
            X=float(T[1])

        if X is None:
            Vals[k]=(P1,P2)
        else:
            Vals[k]=(P1,P2,fOpt(X))

    return Vals


        
