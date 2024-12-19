import copy
import numpy as np

def spectobasis(SM):
    """Converts any spectral matrix to rad/s and cartesian radians"""

    SM = copy.deepcopy(SM)

    Sfac = 1.0
    if 'funit' in SM.keys() and SM['funit'].lower() == 'hz':
        SM['freqs'] *= 2 * np.pi
        Sfac /= (2 * np.pi)

    r2d = np.pi / 180
    if 'dunit' in SM.keys():
        if SM['dunit'][:3].lower() == 'car':
            SM['dirs'] = SM['dirs'] * r2d
            Sfac /= r2d
        elif SM['dunit'][:3].lower() == 'nau':
            if 'xaxisdir' in SM.keys():
                SM['dirs'] += (90 - SM['xaxisdir'])
            SM['dirs'] = r2d * (270 - SM['dirs'])
            Sfac /= r2d

    if 'S' in SM.keys() and (isinstance(SM['S'], np.ndarray) and \
                             SM['S'].size > 0):
        SM['S'] *= Sfac

    return SM, Sfac