"""
Code for figures
"""

# initialize
from IPython.display import display, Markdown
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.gridspec as gridspec
import numpy as np
import numpy.fft as fft
import scipy as sci
from pprint import pprint
from importlib import reload

import pairs as p
reload(p)
import apod as a
reload(a)


plt.style.use("grayscale")
matplotlib.rcParams["axes.facecolor"] = '#f8f8f8'  # light gray in figures
matplotlib.rcParams["figure.facecolor"] = 'white'  # surrounded with white
matplotlib.rcParams["axes.facecolor"] = 'white'  # fig background 
matplotlib.rcParams["axes.spines.left"] =   False  # display axis spines
matplotlib.rcParams["axes.spines.bottom"] = True
matplotlib.rcParams["axes.spines.top"] =    False
matplotlib.rcParams["axes.spines.right"] =  False

Blu = '#3973ac'
Red = '#ff4c1a' #ff751a'
Grin = '#59b300'

for i in ('font.size','axes.labelsize','legend.fontsize','legend.title_fontsize'):
    matplotlib.rcParams[i]=8
for i in ('xtick.labelsize', 'ytick.labelsize'):
    matplotlib.rcParams[i]=8

matplotlib.rcParams["lines.linewidth"] = 1
# plt.rcParams['figure.figsize'] = (180/25.4, 215/25.4)

(fig_width, fig_height) = (180/25.4, 215/25.4) #plt.rcParams['figure.figsize']
#fig_size = [fig_width * 1.5, fig_height ]
fig_size = [fig_width, fig_height ]

def Fig1():
    # let's draw this

    fig, axs = plt.subplots(ncols=2, nrows=1, figsize = [fig_width*0.6, fig_height*0.24], sharey=True, squeeze=True)
    #, title='the complex plane $\mathbb{C}$')
    a = 2
    b = 1.5
    z = a + 1j*b             #  i is noted j in python

    for i in range(2):
        axs[i].plot([-2,3],[0,0],':k') # the real axis
        axs[i].plot([0,0],[-2,3],':k') # the imaginary axis
        axs[i].scatter([1,0,-1,0],[0,1,0,-1], 30)
        axs[i].scatter(z.real, z.imag, 50, color='red')
        axs[i].set_xlabel("Real")
        axs[i].plot([0,z.real],[0,z.imag],'--k')
        axs[i].spines.bottom.set_visible(False)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        # axs[i].spines.left.set_visible(True)
        # axs[i].spines.top.set_visible(True)
        # axs[i].spines.right.set_visible(True)
    axs[0].text(1,0.2,'1')
    axs[0].text(-1,0.2,'-1')
    axs[0].text(0.2,1,'i')
    axs[0].text(0.2,-1,'-i')
    axs[0].set_ylabel("Imaginary")
    axs[0].plot([z.real,z.real],[0,z.imag],':k')
    axs[0].plot([0,z.real],[z.imag,z.imag],':k')
    axs[0].set_xlabel("Real")
    axs[0].set_ylabel("Imaginary")

    axs[0].text(a, -0.4, '$a$')
    axs[0].text(-0.4, b,'$b$')

    axs[0].text(a-0.6,b+0.4,'$z = a +ib$');
    axs[1].text(a-0.6,b+0.4,r'$z = R e^{i \theta}$');

    t = np.linspace(0, np.arctan2(b,a),30)
    axs[1].plot(np.cos(t), np.sin(t))
    axs[1].text(a-0.8, b/2+0.4, '$R$', rotation=35)
    axs[1].text(1.1,0.3, r'$\theta$');
    #fig.suptitle('two notations of complex numbers in the complex plane $\mathbb{C}$');

def Fig2():
#    global fig_width, fig_height
    fig, axs = plt.subplots(ncols=1, nrows=3, figsize=[fig_width, fig_height*0.6] , sharex=True, squeeze=True)

    t = np.arange(0.0, 5, 0.01)
    w0 = np.cos(2 * np.pi * t)
    w1 = np.cos(2.777 * np.pi * t)
    w2 = np.cos(2.03 * np.pi * t)
    w0p = np.cos(2 * np.pi * t - 0.42*np.pi)
    for i,ww in enumerate([w1,w2,w2]):
        if i == 2:  # c/
            ref = w0p
            sig = w2
        else:
            ref = w0
            sig = ww
        prod = ref*sig
        axs[i].plot(t,sig, 'k')
        axs[i].plot(t,ref,'--')
        axs[i].plot(t,prod,':')
        axs[i].fill_between(t,prod, where=(prod>= 0), facecolor=Blu, alpha=1)
        axs[i].fill_between(t,prod, where=(prod<= 0), facecolor=Red, alpha=1)
    axs[0].text(-0.1, 0.9, "a/")
    axs[1].text(-0.1, 0.9, "b/")
    axs[2].text(-0.1, 1.1, "c/")

    #    axs[i].set_xmargin(0.2)
    for i in range(3):
    #    axs[i].annotate("", xy=(5.3, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->",lw=0.5))
        axs[i].plot([0,5.2],[0,0], ls='--')
        axs[i].arrow(5.2, 0, 0.01, 0, lw=1, shape='full',head_width=0.1, head_length=0.1)
        axs[i].spines.bottom.set_visible(False)
        axs[i].set_xticks([])
        axs[i].set_yticks([])

    #axs[1].set_xlabel('time (sec)');
    line = 1.5
    dline = -0.4

# spectral simu
N = 20000 # nb of points
SW = 10000 # Hz
dt = 0.5/SW
t = dt*np.arange(N)
Ai = [3, 6, 0.6]
Ti = [0.03, 0.01, 0.2]
fi = [111, 243, 488]
phi = [0.1, 0.3, 1.5]
colori = [Blu,Grin, Red]
namei = [   '$A:$ slow frequency and medium decay',
            '$B:$ medium frequency and fast decay',
            '$C:$ fast frequency and slow decay']
def locs(i):
    "a single causal resonance"
    return Ai[i]*np.cos(2 * np.pi * fi[i] * t)*np.exp(-t/Ti[i])
def locsn(i):
    "a single signal"
    return Ai[i]*np.cos(2 * np.pi * fi[i] * t + phi[i])

def Fig3():
    # First causal signals
    st = sum([locs(i)  for i in range(3)] )

    Sf = fft.rfft(st).real
    faxe = np.linspace(0,SW,len(Sf))
    def f2i(f):
        "return ~i given f"
        x = f/SW
        i = int(x*len(faxe))
        return i
    fig, axs = plt.subplots(ncols=1, nrows=2, figsize=[fig_width, fig_height*0.5], squeeze=True, layout="constrained")

    # fig, axs = plt.subplots(ncols=1, nrows=4, figsize=[fig_width, fig_height*0.6],
    #     gridspec_kw=dict(width_ratios=[1], height_ratios=[2,2,1.3,1.3], hspace=0.05),
    #     squeeze=True)

    axs[0].plot(t, st, lw=1.5)
    for i in range(3):
        axs[0].plot(t, locs(i)+8+5*i, lw=1, color=colori[i], alpha=1)
        axs[0].text(0.05, 10+5*i, namei[i], color=colori[i]) # 0.05-0.01*i,
    axs[0].set_xlim(xmin=0,xmax=0.2)
    axs[0].set_xlabel('sec')
    axs[0].text(0.05, 2.3, "$s(t)$ sum of $A, B$ and $C$")

    axs[1].plot(faxe, Sf)
    axs[1].set_xlim(xmin=0,xmax=600)
    axs[1].set_xlabel('Hz')
    axs[1].text(20, 300, "$S(f)$")
    for i in range(3):
        b = f2i(fi[i]-30) # begin and end
        e = f2i(fi[i]+30)
        axs[1].plot(faxe[b:e], Sf[b:e], lw=4, color=colori[i], alpha=0.5) #, ls='--' )
        axs[1].text(b, 300, namei[i][0:4], color=colori[i])
    for ax in axs[:2]:
        ax.yaxis.set_visible(False)
    
    # Then non causal
    # stn = sum([locsn(i)  for i in range(3)] )
    # axs[2].plot(t, stn)
    # axs[3].text(0,0,'AX3')
    plt.show()

def Fig4():
    # first parameters
    deltat = 0.0002
    Smax = 0.5/deltat
    Tmax = 3
    timexlim = [-1,1]
    specxlim = [-20,20]
    # axes
    t = np.arange(0, Tmax, deltat)
    N = len(t)
    t = np.concatenate([-t[-1:0:-1], t])
#    print(N, len(t), Smax)
    Freq = 10
    LifeTime = 0.4
    FreqAxis = np.linspace(-Smax, Smax, 2*N-1)
    Fullsignal = np.cos(Freq * 2 * np.pi * t)*np.exp(-abs(t)/LifeTime)
    iFullsignal = np.sin(Freq * 2 * np.pi * t)*np.exp(-abs(t)/LifeTime)
    fig, axs = plt.subplots(ncols=2, nrows=4, figsize=[fig_width, fig_height*0.6], sharex=False, sharey=False, squeeze=True)
    plt.subplots_adjust(hspace=0.05)

    for axr in axs:  # draw axes
        axr[0].plot(timexlim,[0,0], color=(0.1, 0.1, 0.1, 0.3))
        axr[1].plot(specxlim,[0,0], color=(0.1, 0.1, 0.1, 0.3))
    # build and draw signal
    for i,mask in enumerate([0.5, 0.5*np.where(t>=0,1,-1), np.where(t>=0,1,0), np.where(t>=0,1,0)]):
        # print(mask)
        signal = Fullsignal*mask
        isignal = np.zeros_like(Fullsignal)
        if i == 0:
            text = 'even signal'
            y = 0.4
        elif i == 1:
            text = 'odd signal'
            y = 0.4
        elif i == 2:
            text = 'causal signal'
            y = 0.6
        elif i == 3:
            text = 'causal signal'
            isignal = 0.5*iFullsignal*mask
            signal = 0.5*signal + 1j*isignal 
            y = 0.3
            axs[i,0].text(-1, y+0.12 ,"complex")
        axs[i,0].plot(t, isignal, label="imag", color=Red)
        axs[i,0].plot(t, signal.real, label="real",color=Blu)
        axs[i,0].text(-1, y ,text)
        axs[i,0].set_xlim(*timexlim)
        # build and draw spectrum
        Spectrum = np.fft.fftshift(np.fft.fft(np.fft.fftshift(signal)))
#        print(len(signal),len(Spectrum))
        axs[i,1].plot(FreqAxis, Spectrum.imag, color=Red)
        axs[i,1].plot(FreqAxis, Spectrum.real, color=Blu)
        if i==3:
            axs[i,1].plot(FreqAxis, abs(Spectrum), label="magnitude", color='black', ls=':')
        axs[i,1].set_xlim(*specxlim)
        axs[i,1].set_ylim(-1500,1500)   # adpt depending on spectrum parameters...
        for ax in axs[i,:]:
            ax.yaxis.set_visible(False)
            if i<3:
                ax.xaxis.set_visible(False)
                ax.spines.bottom.set_visible(False)
    # titles and legends
    axs[3,0].legend(loc="lower left")
    axs[3,1].legend(loc="lower left")

    axs[0,0].set_title('The signal')
    axs[3,0].set_xlabel('Time')
    axs[0,1].set_title("its FT")
    axs[3,1].set_xlabel('Frequency')

    # f,a = plt.subplots(nrows=2)
    # a[0].plot( Fullsignal ,'b')
    # a[0].plot( np.where(t>=0,1,-1)*iFullsignal ,'r')
    # a[1].plot( np.fft.fftshift(np.fft.fft(np.fft.fftshift(Fullsignal+ 1j*iFullsignal))).real , 'b')
    # a[1].plot( np.fft.fftshift(np.fft.fft(np.fft.fftshift(Fullsignal+ 1j*np.where(t>=0,1,-1)*iFullsignal))).imag, 'r' )
def Fig5():

    fig, axes = plt.subplots(nrows=5, ncols=2, squeeze=False, figsize=[fig_width, fig_height*0.8])
    plt.subplots_adjust(wspace=0.08)

    lig =0
    p.draw(width=2, f=p.cosine, name='pure cosine', ax1=axes[lig,0], ax2=axes[lig,1], modu=False)

    lig += 1
    p.draw(width=1, f=p.gauss, name='Gaussian decay', ax1=axes[lig,0], ax2=axes[lig,1])
    lig += 1
    p.draw(width=0.5, f=p.exp, name='Exponential decay', ax1=axes[lig,0], ax2=axes[lig,1])
    lig += 1
    p.draw(width=100, f=p.gate, name='Gate', ax1=axes[lig,0], ax2=axes[lig,1])
    lig += 1
    p.draw(width=234, f=p.noise, name='white noise', ax1=axes[lig,0], ax2=axes[lig,1], modu=True)

def Fig6():
    raise Exception('Archive - Not used anymore')
    N = 60000 # max nb of points
    SW = 10000 # Hz
    dt = 0.5/SW
    t = dt*np.arange(N)
    Ai = 1.0 
    Ti = 0.2
    fi = 488
    Ni = [1250, 5000, 14140]  # length of truncated lines
    def locs6(i):
        "a single resonance"
        return Ai*np.cos(2 * np.pi * fi * t)*np.exp(-t/Ti)

    st = locs6(0)
    Sf = fft.rfft(st, n=N)

    faxe = np.linspace(0,SW,len(Sf))

    fig = plt.figure(tight_layout=True, figsize=[fig_width, fig_height*0.5])

    gs = gridspec.GridSpec(3, 3)
    plt.subplots_adjust(wspace=0.05)

    axs = {}
    axs[' '] = fig.add_subplot(gs[0, :])

    for i in range(3):
        axs[' '].plot(t[:Ni[i]], st[:Ni[i]]+2.2*i, lw=0.5, color=colori[i])
        axs[' '].plot([0,1], [2.2*i,2.2*i], 'k:', lw=1)
    axs[' '].set_xlabel('sec',labelpad=-10)
    axs[' '].set_xlim(xmin=0, xmax=0.8)

    axs['a/'] = fig.add_subplot(gs[1, 0])
    axs['b/'] = fig.add_subplot(gs[1, 1])
    axs['c/'] = fig.add_subplot(gs[1, 2])

    axs['d/'] = fig.add_subplot(gs[2, 0])
    axs['e/'] = fig.add_subplot(gs[2, 1])
    axs['f/'] = fig.add_subplot(gs[2, 2])

    labels = list(axs.keys())
    for i in range(3):
        # absorptive
        lab = labels[i+1]
        Nii = Ni[i]
        for j in range(3):
            Sf = fft.rfft(st[:Nii], n=N).real
            axs[lab].plot(faxe, Sf/max(Sf), lw=1, color=colori[i])

            Sf = fft.rfft(a.apod_sin(Nii)*st[:Nii], n=N).real
            axs[lab].plot(faxe, Sf/max(Sf), '--', lw=0.7, color=colori[i])
            
        # modulus
        lab = labels[i+4]
        Nii = Ni[i]
        for j in range(3):
            Sf = abs( fft.rfft(st[:Nii], n=N) )
            axs[lab].plot(faxe, Sf/max(Sf), lw=0.7, color=colori[i])

            Sf = abs( fft.rfft(a.apod_sin(Nii)*st[:Nii], n=N) )
            axs[lab].plot(faxe, Sf/max(Sf), '--', lw=0.7, color=colori[i])

            Sf = abs( fft.rfft(a.apod_sin(Nii, maxi=0.5)*st[:Nii], n=N) )
            axs[lab].plot(faxe, Sf/max(Sf), ':', lw=1, color=colori[i])

    # finalize
    axs['a/'].set_xlabel('Hz', labelpad=0, loc="left")
    for i in range(6):
        lab = labels[i+1]
        axs[lab].plot([400,550],[0,0],'k:', lw=1)
        axs[lab].set_xlim(xmin=460,xmax=515)
    #    axs[lab].text(510, 0.9, "$S(f)$")

    for label, ax in axs.items():
        trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            verticalalignment='top')
        ax.yaxis.set_visible(False)
    plt.show()

def Fig7():
    raise Exception('Archive - Not used anymore')
    N = 60000 # max nb of points
    SW = 10000 # Hz
    dt = 0.5/SW
    t = dt*np.arange(N)
    Ai = 1.0 
    Ti = 0.2
    fi = 488
#    Ni = [1250, 5000, 14140]  # length of truncated lines
    def locs7():
        "a single resonance"
        return Ai*np.cos(2 * np.pi * fi * t)*np.exp(-t/Ti)
    st = locs7()+0.3*np.random.randn(N)
    Sf = fft.rfft(st, n=2*N)
    faxe = np.linspace(0,SW,len(Sf))

    fig = plt.figure(tight_layout=True)

    gs = gridspec.GridSpec(2, 3)

    axs = {}
    axs[' '] = fig.add_subplot(gs[0, :])

    for i in range(2):
        axs[' '].plot(t, st, 'k', lw=0.5)
        axs[' '].plot([0,1], [0,0], 'k:', lw=1)
    axs[' '].set_xlabel('sec            ',labelpad=-0)
    axs[' '].set_xlim(xmin=0, xmax=0.8)

    axs['a/'] = fig.add_subplot(gs[1, 0])
    axs['b/'] = fig.add_subplot(gs[1, 1])
    axs['c/'] = fig.add_subplot(gs[1, 2])

    labels = list(axs.keys())
    # absorptive
    lab = labels[i+1]
    zoom=np.arange(2680, 2870)  #2*1800, 2*1910)

    Sf = fft.rfft(st, n=2*N).real
    axs['a/'].plot(faxe, Sf/max(Sf), 'k', lw=1)
    axs['a/'].plot(faxe[zoom], 5*Sf[zoom]/max(Sf)+0.4, 'k', lw=1)
    axs['c/'].plot(faxe, Sf/max(Sf), 'grey', lw=1)

    Sf = fft.rfft(a.apod_em(N, 1.0, SW)*st, n=2*N).real
    axs['b/'].plot(faxe, Sf/max(Sf), 'k', lw=1)
    axs['b/'].plot(faxe[zoom], 5*Sf[zoom]/max(Sf)+0.4, 'k', lw=1)

    Sf = fft.rfft(a.gaussenh(N, 1.0, SW, enhancement=1.0)*st, n=2*N).real
    axs['c/'].plot(faxe, Sf/max(Sf), 'k', lw=1)
    axs['c/'].plot(faxe[zoom], 5*Sf[zoom]/max(Sf)+0.4, 'k', lw=1)
            
    # finalize
    axs['a/'].set_xlabel('Hz', labelpad=0, loc="left")
    for i in range(3):
        lab = labels[i+1]
        axs[lab].plot([400,550],[0,0],'k:', lw=1)
        axs[lab].set_xlim(xmin=440,xmax=505)
        axs[lab].set_ylim(ymin=-0.1, ymax=1.1)
        axs[lab].text(440, 0.5,'x5')
    #    axs[lab].text(510, 0.9, "$S(f)$")

    for label, ax in axs.items():
        trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            verticalalignment='top')
        ax.yaxis.set_visible(False)
    plt.show()

#################################################
def FWHM(x,y,start,end):
    "estimates FWMH of an isolated line"
    valmax = max(y[start:end])
    v2 = valmax/2
    i = start
    while y[i] < v2:
        i += 1
    first = i
    while y[i] > v2:
        i += 1
    last = i
    return x[last]-x[first]


def Figcomb():
    """Fig 6 """
    N = 60000 # max nb of points
    SW = 10000 # Hz
    dt = 0.5/SW
    t = dt*np.arange(N)
    Ai = 1.0 
    Ti = 0.2
    fi = 488
    Ni = [1250, 5000, 14140]  # length of truncated lines
    def locs6(i):
        "a single resonance"
        return Ai*np.cos(2 * np.pi * fi * t)*np.exp(-t/Ti)

    st = locs6(0)
    Sf = fft.rfft(st, n=N)

    faxe = np.linspace(0,SW,len(Sf))

    fig = plt.figure(tight_layout=True, figsize=[fig_width, fig_height])

    gs = gridspec.GridSpec(5, 3)

    axs = {}
    axs[' '] = fig.add_subplot(gs[0, :])

    # FIDS
    for i in range(3):
        axs[' '].plot(t[:Ni[i]], st[:Ni[i]]+2.2*i, lw=0.5, color=colori[i])
        axs[' '].plot([0,1], [2.2*i,2.2*i], 'k:', lw=1)
    axs[' '].set_xlabel('sec',labelpad=-10)
    axs[' '].set_xlim(xmin=0, xmax=0.8)

    # zoom on peak
    axs['a/'] = fig.add_subplot(gs[1, 0])
    axs['b/'] = fig.add_subplot(gs[1, 1])
    axs['c/'] = fig.add_subplot(gs[1, 2])

    axs['d/'] = fig.add_subplot(gs[2, 0])
    axs['e/'] = fig.add_subplot(gs[2, 1])
    axs['f/'] = fig.add_subplot(gs[2, 2])

    labels = list(axs.keys())
    for i in range(3):   # a/ b/ c/
        # absorptive
        # 1400 1550
        lab = labels[i+1]
        Nii = Ni[i]

        Sf = fft.rfft(st[:Nii], n=N).real
        axs[lab].plot(faxe,Sf/max(Sf), lw=1, color=colori[i])
        htxt = round(FWHM(faxe, Sf, 1400, 1550),1)
        axs[lab].text(502, 0.94, htxt, color=colori[i]) #, transform=ax.transAxes + trans, verticalalignment='top')

        Sf = fft.rfft(a.apod_sin(Nii)*st[:Nii], n=N).real
        axs[lab].plot(faxe,Sf/max(Sf), '--', lw=0.7, color=colori[i])
        htxt = round(FWHM(faxe, Sf, 1400, 1550),1)
        axs[lab].text(502, 0.8, htxt, color=colori[i]) #, transform=ax.transAxes + trans, verticalalignment='top')
            
        # modulus
        lab = labels[i+4]  # d/ e/ f/
        Nii = Ni[i]
        Sf = abs( fft.rfft(st[:Nii], n=N) )
        axs[lab].plot(faxe, Sf/max(Sf), lw=0.7, color=colori[i])
        htxt = round(FWHM(faxe, Sf, 1400, 1550),1)
        axs[lab].text(502, 0.94, htxt, color=colori[i]) #, transform=ax.transAxes + trans, verticalalignment='top')

        Sf = abs( fft.rfft(a.apod_sin(Nii)*st[:Nii], n=N) )
        axs[lab].plot(faxe, Sf/max(Sf), '--', lw=0.7, color=colori[i])
        htxt = round(FWHM(faxe, Sf, 1400, 1550),1)
        axs[lab].text(502, 0.8, htxt, color=colori[i]) #, transform=ax.transAxes + trans, verticalalignment='top')

        Sf = abs( fft.rfft(a.apod_sin(Nii, maxi=0.5)*st[:Nii], n=N) )
        axs[lab].plot(faxe, Sf/max(Sf), ':', lw=1, color=colori[i])
        htxt = round(FWHM(faxe, Sf, 1400, 1550),1)
        axs[lab].text(502, 0.66, htxt, color=colori[i]) #, transform=ax.transAxes + trans, verticalalignment='top')

    # finalize
    axs['a/'].set_xlabel('Hz', labelpad=0, loc="left")
    for i in range(6):
        lab = labels[i+1]
        axs[lab].plot([400,550],[0,0],'k:', lw=1)
        axs[lab].set_xlim(xmin=460,xmax=515)
    #    axs[lab].text(510, 0.9, "$S(f)$")

#---------------------------------------------------
    np.random.seed(123)
    st = locs6(0)+0.3*np.random.randn(N)
    Sf = fft.rfft(st, n=2*N)
    faxe = np.linspace(0,SW,len(Sf))

    axs['g/'] = fig.add_subplot(gs[3,:])

    for i in range(2):
        axs['g/'].plot(t, st, 'k', lw=0.5)
        axs['g/'].plot([0,1], [0,0], 'k:', lw=1)
    axs['g/'].set_xlabel('sec            ',labelpad=-0)
    axs['g/'].set_xlim(xmin=0, xmax=0.8)

    axs['h/'] = fig.add_subplot(gs[4, 0])
    axs['i/'] = fig.add_subplot(gs[4, 1])
    axs['j/'] = fig.add_subplot(gs[4, 2])

    labels = list(axs.keys())
    # absorptive
    zoom=np.arange(2680, 2870)  #2*1800, 2*1910)

    Sf = fft.rfft(st, n=2*N).real
    axs['h/'].plot(faxe, Sf/max(Sf), 'k', lw=1)
    axs['h/'].plot(faxe[zoom], 5*Sf[zoom]/max(Sf)+0.4, 'k', lw=1)
    htxt = round(FWHM(faxe, Sf, 2920, 2940),1)
    axs['h/'].text(495, 0.8, htxt) #, transform=ax.transAxes + trans, verticalalignment='top')

#    axs['i/'].plot(faxe, Sf/max(Sf), 'grey', lw=1)

    Sf = fft.rfft(a.apod_em(N, 1.0, SW)*st, n=2*N).real
    axs['i/'].plot(faxe, Sf/max(Sf), 'k', lw=1)
    axs['i/'].plot(faxe[zoom], 5*Sf[zoom]/max(Sf)+0.4, 'k', lw=1)
    htxt = round(FWHM(faxe, Sf, 2920, 2940),1)
    axs['i/'].text(495, 0.8, htxt) #, transform=ax.transAxes + trans, verticalalignment='top')

    Sf = fft.rfft(a.gaussenh(N, 1.0, SW, enhancement=1.0)*st, n=2*N).real
    axs['j/'].plot(faxe, Sf/max(Sf), 'k', lw=1)
    axs['j/'].plot(faxe[zoom], 5*Sf[zoom]/max(Sf)+0.4, 'k', lw=1)
    htxt = round(FWHM(faxe, Sf, 2920, 2940),1)
    axs['j/'].text(495, 0.8, htxt) #, transform=ax.transAxes + trans, verticalalignment='top')
            
    # finalize
    axs['h/'].set_xlabel('Hz', labelpad=0, loc="left")
    for i in range(3):
        lab = labels[i+8]
        axs[lab].plot([400,550],[0,0],'k:', lw=1)
        axs[lab].set_xlim(xmin=440,xmax=505)
        axs[lab].set_ylim(ymin=-0.1, ymax=1.1)
        axs[lab].text(440, 0.5,'x5')
    #    axs[lab].text(510, 0.9, "$S(f)$")

    for label, ax in axs.items():
        trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
        if label == 'g/':
            xoff=0.94
            yoff = 1.1
        else:
            xoff = 0
            yoff = 1
        ax.text(xoff, yoff, label, transform=ax.transAxes + trans,
            verticalalignment='top')
        ax.yaxis.set_visible(False)
    plt.show()
#Fig1()