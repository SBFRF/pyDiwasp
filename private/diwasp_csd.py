import numpy as np
from scipy.signal import csd

def diwasp_csd(x, y, nfft, fs, flag=1):
    """
    Diwasp cross spectral density.
    If flag = 1, use scipy's cross spectral density function
    If flag = 2, use custom cross spectral density function

    [Pxy, f] = diwasp_csd(x,y,nfft,fs)
    """

    if flag == 1:
        f, S = csd(y, x, fs=fs, window='hamming', nperseg=nfft,
                   noverlap=0, nfft=nfft, detrend=False)
    elif flag == 2: # match mlab version if Signal Processing Toolbox not available
        hann = 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, int(nfft / 2) + 1) /
            (nfft + 1)))
        win = np.hstack((hann, np.flipud(hann)))
        nw = np.size(win)
        nseg = int(np.size(x) / nw)
        S = np.zeros(nfft, dtype='complex128')
        for iseg in range(nseg):
            ind = nw * iseg + np.arange(nw)
            xw = win * x[ind]
            yw = win * y[ind]
            Px = np.fft.fft(xw, nfft)
            Py = np.fft.fft(yw, nfft)
            Pxy = Py * np.conj(Px)
            S += Pxy
        nfac = fs * nseg * np.linalg.norm(win) ** 2
        S = np.hstack((S[0], 2 * S[1:int(nfft / 2)], S[int(nfft / 2)])) / nfac
        f = (fs / nfft) * np.arange(int(nfft / 2) + 1).T

    return S, f
