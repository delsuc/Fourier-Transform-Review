####### Apodisations ##########
import numpy as np
def _shifted_apod(N, func,  *arg, maxi=0.5, **kw):
    """
    generic wrapper which allows to shift any apodisation function
    maxi is the location of the maximum point,
        ranges from 0            (max at the beginning )
                to 0.5 - default (max at the center)
    """
    if maxi<0.0 or maxi>0.5:
        raise ValueError("maxi should be within [0...0.5]")
    size = N
    if maxi != 0.5:       # if maxi 0.5 -> same size, if 0 -> double size 
        size = int( size*2*(1-maxi) )
        start = size - N
    else:
        start = 0
#    print(start, size)
    e = func(size, *arg, **kw)
    if start != 0:
        e = e[start:]
    return e

#-------------------------------------------------------------------------------
def kaiser(N, beta, maxi=0.5):
    """
    apply a Kaiser apodisation
    beta is a positive number
    maxi is the location of the maximum point,
        ranges from 0            (max at the beginning )
                to 0.5 - default (max at the center)

    Kaiser is a versatile general apodisation method.
    with maxi = 0.5 (default - means the function is centered) it approximates classical functions: 
            useful for modulus spectra 
        beta    Window shape
        ----    ------------
        0       Rectangular
        5       Similar to a Hamming
        6       Similar to a Hanning
        8.6     Similar to a Blackman

    with maxi<0.5 it allows to generate an apodisation very close to gaussenh()
        shaping precisely the maximum and the curvature indepently
    for instance for a dataset of 10kHz spectral width and 64k points
    kaiser(maxi=0.15, beta=15) and gaussenh(0.5, enhancement=1)  are similar

    can also be sued to simulate the Kilgour apodisation
    """
    return _shifted_apod(N, np.kaiser,  beta, maxi=maxi)
#-------------------------------------------------------------------------------
def hamming(N, maxi=0.5):
    """
    apply a Hamming apodisation
    maxi is the location of the maximum point,
        ranges from 0            (max at the beginning )
                to 0.5 - default (max at the center)
    """
    return _shifted_apod(N, np.hamming, maxi=maxi)
#-------------------------------------------------------------------------------
def hanning(N, maxi=0.5):
    """
    apply a Hanning apodisation
    maxi is the location of the maximum point,
        ranges from 0            (max at the beginning )
                to 0.5 - default (max at the center)
    """
    return _shifted_apod(N, np.hanning, maxi=maxi)
#-------------------------------------------------------------------------------
def apod_gm(N, gb, specwidth):
    """
    apply an gaussian apodisation, gb is in Hz
    WARNING : different from common definition of gaussian enhancement
                if this is what you need, check the plugin gaussenh() 
    """
    sw = specwidth
    size = N
    e = np.exp(-(gb*np.arange(size)/sw)**2)
    return e

def gaussenh(N, width, specwidth, enhancement=2.0):
    """
    apply an gaussian enhancement, width is in Hz
    enhancement is the strength of the effect
    multiplies by gauss(width) * exp(-enhancement*width)
    """
    sw = specwidth
    size = N
    baseax = width*np.arange(size)/sw  # t/Tau
    e = np.exp(enhancement*baseax)
    e *= np.exp(-(baseax)**2)
    e *= 1.0/np.max(e)          # normalize
    return e

#-------------------------------------------------------------------------------
def apod_tm(N, tm1, tm2 ):
    """
    apply a trapezoide apodisation, lb is in Hz
    WARNING : different from common definition of apodisation
    This commands applies a trapezoid filter function to the data-
    set. The function raises from 0.0 to 1.0 from the first point to 
    point tm1. The function then stays to 1.0 until point tm2, from which 
    it goes down to 0.0 at the last point.
    if tm2 = -1 nothing is done
    """
    size = N
    ftm1 = tm1
    ftm2 = size-tm2
    e = np.ones(size)
    e[0:ftm1]  = np.linspace(0, 1, ftm1)
    if tm2 != -1:
        e[tm2:]    = np.linspace(1, 0, ftm2)
    return e
#-------------------------------------------------------------------------------
def apod_em(N, lb, specwidth):
    """
    apply an exponential apodisation, lb is in Hz
    WARNING : different from common definition of apodisation
    """
    sw = specwidth
    size = N
    e = np.exp(-lb*np.arange(size)/sw)
    return  e
#-------------------------------------------------------------------------------
def apod_sq_sin(N, maxi=0.0):
    """
    apply a squared sinebell apodisation
    maxi ranges from 0 to 0.5
    """
    import math as m
    if maxi<0.0 or maxi>0.5:
        raise ValueError("maxi should be within [0...0.5]")
    # compute shape parameters
    size = N
    s = 2*(1-maxi)
    zz = m.pi/((size-1)*s)
    yy = m.pi*(s-1)/s         #yy has the dimension of a phase
    # then draw buffer
    e = np.sin( zz*np.arange(size)+yy)**2
    return e
    
#-------------------------------------------------------------------------------
def _apod_sin(N, maxi=0.0):
    """
    apply a sinebell apodisation
    maxi is the location of the maximum point,
            ranges from 0    (max at the beginning == a cosine func)
                    to 0.5  (max at the center == a sine func)
    """
    import math as m
    if maxi<0.0 or maxi>0.5:
        raise ValueError("maxi should be within [0...0.5]")
    # compute shape parameters
    size = N
    s = 2*(1-maxi)
    zz = m.pi/((size-1)*s)
    yy = m.pi*(s-1)/s         #yy has the dimension of a phase
    # then draw buffer
    e = np.sin( zz*np.arange(size)+yy)
    return e

def apod_sin(N, maxi=0.0):
    """
    apply a sinebell apodisation
    maxi is the location of the maximum point,
            ranges from 0    (max at the beginning == a cosine func)
                    to 0.5  (max at the center == a sine func)
    """
    import math as m
    size = N
    def lsin(M):
        return np.sin(np.linspace(0,np.pi,M))
    e = _shifted_apod(N, lsin, maxi=maxi)
    return e
    