import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


D00=np.load("Data\PlaneC2H4_0-CAM_0.1_0.0_0-1,0.8.npz")
D30=np.load("Data\PlaneC2H4_30-CAM_0.1_0.0_0-1,0.8.npz")
D80=np.load("Data\PlaneC2H4_80-CAM_0.1_0.0_0-1,0.8.npz")


x=D00['x']
Out=D00['Data']-D80['Data']


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

mx,my=np.meshgrid(x,x)
surf=ax.plot_surface(mx,my,Out,
                     rstride=1,cstride=1,
                     zorder=0)

plt.show()
