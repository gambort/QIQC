RawElementData="""1,1.008,Hydrogen,H
2,4.003,Helium,He
3,6.941,Lithium,Li
4,9.012,Beryllium,Be
5,10.811,Boron,B
6,12.011,Carbon,C
7,14.007,Nitrogen,N
8,15.999,Oxygen,O
9,18.998,Fluorine,F
10,20.18,Neon,Ne
11,22.99,Sodium,Na
12,24.305,Magnesium,Mg
13,26.982,Aluminum,Al
14,28.086,Silicon,Si
15,30.974,Phosphorus,P
16,32.065,Sulfur,S
17,35.453,Chlorine,Cl
18,39.948,Argon,Ar
19,39.098,Potassium,K
20,40.078,Calcium,Ca
21,44.956,Scandium,Sc
22,47.867,Titanium,Ti
23,50.942,Vanadium,V
24,51.996,Chromium,Cr
25,54.938,Manganese,Mn
26,55.845,Iron,Fe
27,58.933,Cobalt,Co
28,58.693,Nickel,Ni
29,63.546,Copper,Cu
30,65.39,Zinc,Zn
31,69.723,Gallium,Ga
32,72.64,Germanium,Ge
33,74.922,Arsenic,As
34,78.96,Selenium,Se
35,79.904,Bromine,Br
36,83.8,Krypton,Kr
37,85.468,Rubidium,Rb
38,87.62,Strontium,Sr
39,88.906,Yttrium,Y
40,91.224,Zirconium,Zr
41,92.906,Niobium,Nb
42,95.94,Molybdenum,Mo
43,98,Technetium,Tc
44,101.07,Ruthenium,Ru
45,102.906,Rhodium,Rh
46,106.42,Palladium,Pd
47,107.868,Silver,Ag
48,112.411,Cadmium,Cd
49,114.818,Indium,In
50,118.71,Tin,Sn
51,121.76,Antimony,Sb
52,127.6,Tellurium,Te
53,126.905,Iodine,I
54,131.293,Xenon,Xe
55,132.906,Cesium,Cs
56,137.327,Barium,Ba
57,138.906,Lanthanum,La
58,140.116,Cerium,Ce
59,140.908,Praseodymium,Pr
60,144.24,Neodymium,Nd
61,145,Promethium,Pm
62,150.36,Samarium,Sm
63,151.964,Europium,Eu
64,157.25,Gadolinium,Gd
65,158.925,Terbium,Tb
66,162.5,Dysprosium,Dy
67,164.93,Holmium,Ho
68,167.259,Erbium,Er
69,168.934,Thulium,Tm
70,173.04,Ytterbium,Yb
71,174.967,Lutetium,Lu
72,178.49,Hafnium,Hf
73,180.948,Tantalum,Ta
74,183.84,Tungsten,W
75,186.207,Rhenium,Re
76,190.23,Osmium,Os
77,192.217,Iridium,Ir
78,195.078,Platinum,Pt
79,196.967,Gold,Au
80,200.59,Mercury,Hg
81,204.383,Thallium,Tl
82,207.2,Lead,Pb
83,208.98,Bismuth,Bi
84,209,Polonium,Po
85,210,Astatine,At
86,222,Radon,Rn
87,223,Francium,Fr
88,226,Radium,Ra
89,227,Actinium,Ac
90,232.038,Thorium,Th
91,231.036,Protactinium,Pa
92,238.029,Uranium,U
93,237,Neptunium,Np
94,244,Plutonium,Pu
95,243,Americium,Am
96,247,Curium,Cm
97,247,Berkelium,Bk
98,251,Californium,Cf
99,252,Einsteinium,Es
100,257,Fermium,Fm
101,258,Mendelevium,Md
102,259,Nobelium,No
103,262,Lawrencium,Lr
104,261,Rutherfordium,Rf
105,262,Dubnium,Db
106,266,Seaborgium,Sg
107,264,Bohrium,Bh
108,277,Hassium,Hs
109,268,Meitnerium,Mt
"""

class QuickElements:
    def __init__(self):
        T=RawElementData.split("\n")
        self.ByZ={}
        self.ByID={}
        for L in T:
            D=L.split(",")
            if len(D)<4:
                continue
            Z=int(D[0])
            M=float(D[1])
            Name=D[2]
            ID=D[3]

            Res={'Z':Z,'M':M,'Name':Name,'ID':ID}

            self.ByZ[Z]=Res
            self.ByID[ID]=Res

    def GetZ(self,ID):
        return self.ByID[ID]['Z']

    def GetID(self,Z):
        return self.ByZ[int(Z)]['ID']
            
