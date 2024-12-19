import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend
from interpspec import interpspec
from infospec import infospec
from writespec import writespec
from plotspec import plotspec
from private.velx import velx
from private.vely import vely
from private.pres import pres
from private.elev import elev
from private.wavenumber import wavenumber
from private.IMLM import IMLM
from private.EMEP import EMEP
from private.smoothspec import smoothspec
from private.diwasp_csd import diwasp_csd
from private.check_data import check_data

def dirspec(ID, SM, EP, Options_=None):
    """
    DIWASP V1.4 function
    dirspec: main spectral estimation routine

    [SMout,EPout]=dirspec(ID,SM,EP,{options})

    Outputs:
    SMout	    A spectral matrix structure containing the results
    Epout		The estimation parameters structure with the values actually used for the computation including any default settings.

    Inputs:
    ID			An instrument data structure containing the measured data
    SM   		A spectral matrix structure; data in field SM.S is ignored.
    EP		    The estimation parameters structure. For default values enter EP as []
    [options]  options entered as cell array with parameter/value pairs: e.g.{'MESSAGE',1,'PLOTTYPE',2};
                    Available options with default values:
                        'MESSAGE',1,    Level of screen display: 0,1,2 (increasing output)
                        'PLOTTYPE',1,   Plot type: 0 none, 1 3d surface, 2 polar type plot, 3 3d surface(compass angles), 4 polar plot(compass angles)
                        'FILEOUT',''  	 Filename for output file: empty string means no file output

    Input structures ID and SM are required. Either [EP] or [options] can be included but must be in order if both are included.
    "help data_structures" for information on the DIWASP data structures

    All of the implemented calculation algorithms are as described by:
    Hashimoto,N. 1997 "Analysis of the directional wave spectrum from field data"
    In: Advances in Coastal Engineering Vol.3. Ed:Liu,P.L-F. Pub:World Scientific,Singapore


    Original copyright (C) 2002 Coastal Oceanography Group, CWR, UWA, Perth

    Translated by Chuan Li and Spicer Bak,
    Field Research Facility, US Army Corps of Engineers
    """

    Options = {'MESSAGE':1, 'PLOTTYPE':1, 'FILEOUT':''}

    if Options_ is not None:
        nopts = len(Options_)
    else:
        nopts = 0

    ID = check_data(ID, 1)
    if len(ID) == 0:
        return [], []
    SM = check_data(SM, 2)
    if len(SM) == 0:
        return [], []
    EP = check_data(EP, 3)
    if len(EP) == 0:
        return [], []

    if nopts != 0:
        if nopts % 2 != 0:
            warnings.warn('Options must be in Name/Value pairs - setting to '
                'defaults')
        else:
            for i in range(int(nopts / 2)):
                arg = Options_[2 * i + 1]
                field = Options_[2 * i]
                Options[field] = arg

    ptype = Options['PLOTTYPE']
    displ = Options['MESSAGE']


    print('\ncalculating.....\n\ncross power spectra')

    data = detrend(ID['data'], axis=0)
    ndat, szd = np.shape(ID['data'])

    #get resolution of FFT - if not specified, calculate a sensible value
    if 'nfft' not in EP:
        nfft = int(2 ** (8 + np.round(np.log2(ID['fs']))))
        EP['nfft'] = nfft
    else:
        nfft = int(EP['nfft'])
    if nfft > ndat:
        raise Exception('Data length of {} too small'.format(ndat))

    #calculate the cross-power spectra
    xps = np.empty((szd, szd, int(nfft / 2)), 'complex128')
    for m in range(szd):
        for n in range(szd):
            xpstmp, Ftmp = diwasp_csd(data[:, m], data[:, n],
                                      nfft, ID['fs'], flag=2)
            xps[m, n, :] = xpstmp[1:int(nfft / 2) + 1]
    F = Ftmp[1:int(nfft / 2) + 1]
    nf = int(nfft / 2)
    print('wavenumbers')
    wns = wavenumber(2  * np.pi * F, ID['depth'] * np.ones(np.shape(F)))
    pidirs = np.linspace(-np.pi, np.pi - 2 * np.pi / EP['dres'],
        num=EP['dres'])

    #calculate transfer parameters
    print('transfer parameters\n')
    trm = np.empty((szd, nf, len(pidirs)))
    kx = np.empty((szd, szd, nf, len(pidirs)))
    for m in range(szd):
        trm[m, :, :] = eval(ID['datatypes'][m])(2 * np.pi * F, pidirs, wns,
            ID['layout'][2, m], ID['depth'])
        for n in range(szd):
            kx[m, n, :, :] = wns[:, np.newaxis] * ((ID['layout'][0, n] -
                ID['layout'][0, m]) * np.cos(pidirs) + (ID['layout'][1, n] -
                ID['layout'][1, m]) * np.sin(pidirs))

    Ss = np.empty((szd, nf), dtype='complex128')
    for m in range(szd):
        tfn = trm[m, :, :]
        Sxps = xps[m, m, :]
        Ss[m, :] = Sxps / (np.max(tfn, axis=1) * np.conj(np.max(tfn, axis=1)))

    ffs = np.logical_and(F >= np.min(SM['freqs']), F <= np.max(SM['freqs']))
    SM1 = dict()
    SM1['freqs'] = F[ffs]
    SM1['funit'] = 'Hz'
    SM1['dirs'] = pidirs
    SM1['dunit'] = 'rad'

    # call appropriate estimation function
    print('directional spectra using {} method'.format(EP['method']))
    SM1['S'] = eval(EP['method'])(xps[:, :, ffs], trm[:, ffs, :],
        kx[:, :, ffs, :], Ss[:, ffs], pidirs, EP['iter'], displ)
    SM1['S'][np.logical_or(np.isnan(SM1['S']), SM1['S'] < 0)] = 0

    #Interpolate onto user specified matrix
    print('\ninterpolating onto specified matrix...\n')
    SMout = interpspec(SM1, SM, method='linear')

    #smooth spectrum
    if EP['smooth'].upper() == 'ON':
        print('\nsmoothing spectrum...\n')
        SMout = smoothspec(SMout, [[1, 0.5, 0.25], [1, 0.5, 0.25]])

    infospec(SMout)

    #write out spectrum matrix in DIWASP format
    filename = Options['FILEOUT']
    if len(filename) > 0:
        print('writing out spectrum matrix to file')
        writespec(SMout,filename)

    #plot spectrum
    if ptype > 0:
        print('finished...plotting spectrum')
        plotspec(SMout, ptype)
        T = 'Directional spectrum estimate using {} method'.format(EP['method'])
        plt.title(T)
        plt.show()

    return SMout, EP
