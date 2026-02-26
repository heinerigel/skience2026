import cmath
import math
from scipy import optimize
import numpy as np

# Based on Yu et al. https://doi.org/10.1016/j.csndt.2014.11.001

def compute_qs_dispersion(d=0.55, E=7.3e9, p_ice=917, sigma=0.33, freq=None):
    """
    Compute QS dispersion curves for ice layer.
    
    Parameters:
    -----------
    d : float
        Layer thickness (default: 0.55)
    E : float
        Young's modulus (default: 7.3e9)
    p_ice : float
        Ice density (default: 917)
    sigma : float
        Poisson's ratio (default: 0.33)
    freq : array-like, optional
        Frequency array. If None, uses linspace(0.5, 5000, 10000)
    
    Returns:
    --------
    dict with keys: 'freq', 'v', 'w', 'k', 'u'
        'freq': frequency array
        'v': phase velocity array (complex)
        'w': angular frequency array
        'k': wavenumber array
        'u': group velocity array
    """
    if freq is None:
        freq = np.linspace(0.5, 5000, 10000)
    freq = np.asarray(freq, dtype=float)
    
    S = E / (2 * (1 + sigma))
    
    def calc_speed1(c, f):
        w = 2 * np.pi * f
        k = w / c
        cl = math.sqrt(E * (1 - sigma) / (p_ice * (1 + sigma) * (1 - 2 * sigma)))
        cs = math.sqrt(E / (2 * p_ice * (1 + sigma)))
        cw = 1430
        
        ks2 = (w**2 / cs**2 - k**2)
        kl2 = (w**2 / cl**2 - k**2)
        
        ks = cmath.sqrt(ks2)
        kl = cmath.sqrt(kl2)
        kw = cmath.sqrt(w**2 / cw**2 - k**2)
        
        gl = cmath.exp(1j * kl * d)
        gs = cmath.exp(1j * ks * d)
        
        a11 = ks2 - k**2
        a12 = ks2 - k**2
        a13 = -2 * ks * k
        a14 = 2 * ks * k
        a15 = 0
        
        a21 = 2 * kl * k
        a22 = -2 * kl * k
        a23 = ks2 - k**2
        a24 = ks2 - k**2
        a25 = 0
        
        a31 = (ks2 - k**2) * gl
        a32 = (ks2 - k**2) / gl
        a33 = -2 * ks * k * gs
        a34 = 2 * ks * k / gs
        a35 = w**2 * 1000 / S
        
        a41 = 2 * kl * k * gl
        a42 = -2 * kl * k / gl
        a43 = (ks2 - k**2) * gs
        a44 = (ks2 - k**2) / gs
        a45 = 0
        
        a51 = kl * gl
        a52 = -kl / gl
        a53 = -k * gs
        a54 = -k / gs
        a55 = kw
        
        aa = np.array([[a11, a12, a13, a14, a15],
                       [a21, a22, a23, a24, a25],
                       [a31, a32, a33, a34, a35],
                       [a41, a42, a43, a44, a45],
                       [a51, a52, a53, a54, a55]], dtype=complex)
        return np.linalg.det(aa)
    
    v = np.zeros(len(freq), dtype=complex)
    
    for i in range(len(freq)):
        f = freq[i]
        try:
            if i == 0:
                v[i] = optimize.newton(lambda c: calc_speed1(c, f), x0=20)
            else:
                v[i] = optimize.newton(lambda c: calc_speed1(c, f), x0=float(np.real(v[i-1])))
        except:
            v[i] = np.nan + 0j
    
    w = np.zeros(len(v))
    k = np.zeros(len(v))
    u = np.zeros(len(v))
    
    for i in range(len(freq)):
        w[i] = 2 * np.pi * freq[i]
        k[i] = w[i] / np.real(v[i])
    
    index = np.arange(1, len(freq) - 1, 1)
    for i in index:
        u[i] = (w[i] - w[i-1]) / (k[i] - k[i-1])
    
    return {'freq': freq, 'v': v, 'w': w, 'k': k, 'u': u}