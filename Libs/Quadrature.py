import numpy as np
from numpy import cos,sin,pi
from scipy.linalg import eig
from scipy.special import gamma, gammaln
from scipy.interpolate import lagrange
from numpy.polynomial.hermite import hermgauss

def QuadCoeffs(x, LHS=False):
    """w=QuadCoeffs(x, LHS=False)
    
    return second order quadrature weights for
    abscissae x. If LHS is true extend to x=0
    """
    
    dx=np.diff( x )
    w=0.5*( np.concatenate( (np.array([0.0]),dx) )
            + np.concatenate( (dx,np.array([0.0]) ) ) )

    if LHS:
        w[0]+=x[0]

    return w

def Mirror(x,wx):
    xp=np.concatenate((-np.flipud(x),x))
    wxp=np.concatenate((np.flipud(wx),wx))

    return xp,wxp

def ClenCurtisWeights(N,S=1.0,Trim=True):
    """x,w=ClenCurtisWeights(N,[s=1.0])

    Calculates Clenshaw-Curtis abcissae and weights
    for integration."""


    theta=pi/2-pi*( np.arange(0,N)+0.5 )/(2.*N)
    x=cos(theta)/sin(theta)
    w=1.0/sin(theta)**2 *pi/N

    return x/S, w/S

def GaussLegendreWeights(n,a=-1,b=1):
    return GaussLegWeights(n,a=a,b=b)
def GaussLegWeights(n,a=-1,b=1):
    """Generates the abscissa and weights for a Gauss-Legendre quadrature.
    between a and b"""

    x,w=np.polynomial.legendre.leggauss(n)
    S=(b-a)/2.
    return (x+1)*S+a,w*S


def OldGaussLegWeights(n,a=-1,b=1):
    """Generates the abscissa and weights for a Gauss-Legendre quadrature.
    % Reference:  Numerical Recipes in Fortran 77, Cornell press."""
    
    x = np.zeros( (n) ) # Preallocations.
    w = x*1.0
    m = (n+1)/2
    for ii in range(1,m+1):
        z = cos(pi*(ii-.25)/(n+.5)) # Initial estimate.
        z1 = z+1
        while (abs(z-z1)>1e-8):
            p1 = 1.0
            p2 = 0.0
            for jj in range(1,n+1):
                p3 = p2
                p2 = p1
                p1 = ((2*jj-1)*z*p2-(jj-1)*p3)/jj   # The Legendre polynomial.
            # end
            pp = n*(z*p1-p2)/(z**2-1)       # The L.P. derivative.
            z1 = z
            z = z1-p1/pp
        # end
        x[ii-1] = -z              # Build up the abscissas.
        x[n-ii] = z
        w[ii-1] = 2/((1-z**2)*(pp**2))          #Build up the weights.
        w[n-ii] = w[ii-1]

    S=(b-a)/2.0
    x = a+(x+1)*S
    w = w*S
    return x,w


def GaussLagWeights(n,alpha=0.0, S=1.0):
    """x,ww,w=GaussLagWeights(n, alpha, [S=1.0])

    \int_0^{\infty} e^{-x}f(x) d x
    \approx \sum ww_i f(x_i) e^{-x_i} = \sum w_i f(x_i)

    [[slightly different if alpha!=0.0]]
    """

    ii=1.0*np.array( range(1,n+1) )
    a=(2.0*ii-1)+alpha
    b=np.sqrt( ii[0:(n-1)]*(ii[0:(n-1)] + alpha) )
    CM=np.diag(a) + np.diag(b,1) + np.diag(b,-1)

    L,V=eig(1.0*CM)

    ind=np.argsort(L)
    x=L[ind].real
    V=V[:,ind].T

    w=gamma(alpha+1.0)*V[:,0]**2
    lw=( gammaln(alpha+1.0) + np.log(V[:,0]**2) )
    w=np.exp(lw)
    lw=lw.real
    ww=np.exp(x+lw)

    return S*x,S*ww,S*w


def GaussHermiteWeights(n,S=1.0, positive=False):
    return GaussHerWeights(n,S,positive)

def GaussHerWeights(n,S=1.0, positive=False):
    """
    """
    if (positive):
        y,wy=hermgauss(2*n)
        y=y[n:]
        wwy=wy[n:]
    else:
        y,wwy=hermgauss(n)

    wy=np.exp(np.log(wwy)+y**2)

    return S*y,S*wy,S*wwy
