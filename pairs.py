# some hard coding of the functions
import numpy as np
from numpy import fft
import matplotlib as mpl
import matplotlib.pylab as plt

Blu = '#3973ac'
Red = '#ff751a'
Grin = '#59b300'


x = np.linspace(0,10,1000)    # a vector of 1000 points equi distant from 0.0 to 10.0

def gate(width=10):
    "return a gate function over x"
    r = np.zeros_like(x)
    r[:width] = 1.0
#    r[-width:] = 1.0
    return r
def gauss(width=1.0):
    "return a centered gaussian function over x"
    r = np.exp(-((x-5)/width)**2)
    r[:500] = 0
    return fft.fftshift(r)
def exp(width=1.0):
    "return a centered exp function over x"
    r = np.exp(-abs(x-5)/width)
    r[:500] = 0
    return fft.fftshift(r)
def noise(width):
    np.random.seed(width)
    return np.random.randn(len(x))
def cosine(width):
    "the delta function"
    r = np.cos(2*np.pi*width*x)
    return r
def position(width):
    "the delta function"
    r = np.zeros_like(x)
    r[int(width*100)] = 1.0
    return r
def draw(width, f, name, ax1, ax2, modu=True):
    "builds the nice drawing"
#    fig, (ax1,ax2) = plt.subplots(ncols=2)
    y = f(width=width)
    xax = np.linspace(-5,5,1000)
    yax = np.linspace(-50,50,1000)
    YY = fft.fftshift(fft.fft(y))
    ax1.plot(xax, fft.fftshift(y), label=name)
    ax1.legend(loc=1)
    if f != cosine:
        ax1.set_xlim(xmin=-0.05)
    if f == cosine:
        YY = np.zeros_like(x)
        YY[500+int(width*10)] = 1.0
        YY[500-int(width*10)] = 1.0

        ax1.plot([-5,5],[0,0], ls='--')
#        ax1.arrow(5.2, 0, 0.01, 0, lw=1, shape='full',head_width=0.1, head_length=0.1)

    ax2.plot(yax, YY.real, label='FT('+name+')', color=Blu)
    if modu: #f == gate:
        ax2.plot(yax, abs(YY), 'k--')
    if f in (gauss, exp, gate):
        ax2.plot(yax, YY.imag, label='FT('+name+')',  color=Red)
    if f == noise:
        ax2.set_xlim(xmin=0)
    else:
        ax2.set_xlim(xmin=-5, xmax=5)
    for ax in [ax1,ax2]:
        for s in ["left", "top", "right"]:
            ax.spines[s].set_visible(False)
            ax.yaxis.set_visible(False)



