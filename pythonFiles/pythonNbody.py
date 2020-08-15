import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mplEasyAnimate import animation
from tqdm import tqdm
from scipy.special import comb
from scipy.special import factorial

G = 6.6e-11

def rk4(f, y0, t, h, ID, pList, massList):
    k1 = h*f(y0, t, ID, pList, massList)
    k2 = h*f(y0+k1/2, t+(h/2), ID, pList, massList)
    k3 = h*f(y0+k2/2, t+(h/2), ID, pList, massList)
    k4 = h*f(y0+k3, t+h, ID, pList, massList)
    return y0 + (k1/6)+(k2/3)+(k3/3)+(k4/6)

def plot_system(state):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(state[:, 0], state[:, 1], 'o')
    ax.quiver(state[:, 0], state[:, 1], state[:, 3], state[:, 4])
    return fig, ax

def nbody(I0, t, ID, pList, massList):
    dydt = np.zeros(6)
    dydt[:3] = I0[3:]
    m = massList[ID]
    
    for i, particle in enumerate(pList):
        if ID != i:
            r = particle[:3] - I0[:3]
            rmag = np.sqrt(sum([x**2 for x in r]))
            if rmag > 0.5:
                rhat = r/rmag
                FMag = (G*massList[i])/((rmag)**2)
                dydt[3:] += FMag*rhat
    #             print('Force on particle {} by {}: {}'.format(ID, i, dydt[3:]))

    return dydt

def int_n_model(model, method, y0, h, massList, t0=0, tf=1):
    ts = np.arange(t0, tf, h)
    ys = np.zeros(shape=(len(ts)+1, y0.shape[0], y0.shape[1]))
    ys[0] = y0
    for i, t in tqdm(enumerate(ts), total=len(ts)):
        for ID, particle in enumerate(ys[i]):
            ys[i+1][ID] = method(model, particle, t, h, ID, ys[i], massList)
    return np.arange(t0, tf+h, h), ys

def getEnergy(y, massList, energy_function):
    energy = np.zeros(shape=(y.shape[0], y.shape[1]))
    for TID, ys in enumerate(y): # Each time step
        for PID, (particle, m) in enumerate(zip(ys, massList)):
            R = particle[:3]
            for oPID, (oParticle, oM) in enumerate(zip(ys, massList)):
                if PID != oPID:
                    r = oParticle[:3]-R
                    rmag = np.sqrt(sum([x**2 for x in r]))
                    energy[TID, PID] += energy_function(rmag, m, oM)
            v = particle[3:]
            vmag = np.sqrt(sum([x**2 for x in v]))
            energy[TID, PID] += (1/2)*m*vmag**2
    return energy

def isBinaryUnbound(y, massList, offset=0):
    energy = getEnergy(y[:, offset:offset+2, :], massList[offset:offset+2], lambda r, m, M: -G*m*M/r)
    if energy[-1, 0] > 0 and energy[-1, 1] > 0:
        return True
    else:
        return False


