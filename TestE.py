import numpy as np
import matplotlib.pyplot as plt

def E(x):
    y=0.5*(1.+np.sqrt(1.-x**2))
    return -y*np.log2(y) - (1.-y)*np.log2(1.-y)

x=np.linspace(1e-5,1.-1e-5,101)

plt.plot(x, E(x), "-",
         linewidth=4)
plt.show()

