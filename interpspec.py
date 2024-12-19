import numpy as np
from warnings import warn
from scipy.interpolate import griddata
from private.hsig import hsig
from private.spectobasis import spectobasis

def interpspec(SMin, SMout, method='linear'):
    """
    DIWASP V1.4 function
    interpspec: interpolates between spectral matrix bases

    SMout=interpspec(SMin,SMout)

    Outputs:
    SMout		Output spectral matrix structure with interpolated power density

    Inputs:
    SMin   	A spectral matrix structure containing the original spectra
    SMout      A spectral matrix defining the new spectral matrix

    SMout only needs to have the frequency and directional axes defined -
    spectral density ignored

    "help data_structures" for information on the DIWASP data structures
    """
    Hs1 = hsig(SMin)

    SMin, facin = spectobasis(SMin)
    SMtmp, facout = spectobasis(SMout)

    s1 = SMin['freqs'][:, np.newaxis] * np.sin(SMin['dirs'])
    c1 = SMin['freqs'][:, np.newaxis] * np.cos(SMin['dirs'])
    s2 = SMtmp['freqs'][:, np.newaxis] * np.sin(SMtmp['dirs'])
    c2 = SMtmp['freqs'][:, np.newaxis] * np.cos(SMtmp['dirs'])

    if np.array_equal(s1, s2) and np.array_equal(c1, c2):
        warn('No interpolation required, skipping')
        Stmp = SMin['S']
    else:
        Stmp = griddata((s1.flatten(), c1.flatten()), SMin['S'].flatten(),
            (s2.flatten(), c2.flatten()), method=method).reshape(s2.shape)

    Stmp[np.isnan(Stmp)] = 0
    SMout['S'] = Stmp / facout

    # check Hsig of mapped spectrum and check sufficiently close to original
    Hs2 = hsig(SMout)
    if (Hs2 - Hs1) / Hs1 > 0.02:
        warn('User defined grid may be too coarse; try increasing' +
            ' resolution of ''SM[\'freqs\']'' or ''SM[\'dirs\']''')

    return SMout