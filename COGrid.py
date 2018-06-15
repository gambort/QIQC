import numpy as np
import numpy.polynomial.legendre as le

x,w=le.leggauss(6)
theta=x[3:]*np.pi
wtheta=w[3:]

z=np.linspace(-1.12*4,1.12*4,(112+1))
h=z[1]-z[0]
r=np.linspace(0.,1.12,(112/8)+1)+h/2.

np.savez("CO.grid.npz", theta=theta, wtheta=wtheta, z=z, r=r)

xyz=np.zeros((len(theta)*len(r)*len(z),3))
I=0
for j in range(len(theta)):
    for k in range(len(r)):
        x=r[k]*np.cos(theta[j])
        y=r[k]*np.sin(theta[j])

        for l in range(len(z)):
            #print "%10.6f %10.6f %10.6f"%(x,y,z[l])
            xyz[I,:]=x,y,z[l]
            I+=1

np.savetxt("CO.grid", xyz, fmt="%11.7f")
