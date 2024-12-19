import warnings
import numpy as np

def check_data(DDS, type_):
    """
    internal DIWASP1.1 function
    checks data structures

      DDS=check_data(DDS,type)
          DDS:  the data structure
          type: 1, Instrument data structure;
                2, Spectral matrix structure;
                3, Estimation parameters structure.

    Updated on 29/04/2013 by r.guedes to fix some errors that were not being
    detected (e.g., ID.datatype not being a cell array).
    """

    #--------------------------------------------------------------------------
    # Defaults
    #--------------------------------------------------------------------------
    SM = dict()
    EP = dict()
    SM['xaxisdir'] = 90
    EP['dres'] = 180
    EP['nfft'] = []
    EP['method'] = 'IMLM'
    EP['iter'] = 100
    error = ''

    #--------------------------------------------------------------------------
    # Instrument data structure
    #--------------------------------------------------------------------------
    if type_ == 1:
        if type(DDS) != dict:
            print('DIWASP data_check: Instrument data type is not a '
                'dictionary')
        nc = 1
        if 'layout' in DDS:
            nr, nc = np.shape(DDS['layout'])
            if nr < 3:
                if nr == 2:
                    np.array(DDS['layout'])[2, :] = 0
                else:
                    error = 'layout'
    
        if ('datatypes' not in DDS or type(DDS['datatypes']) not in 
            (list, np.ndarray) or np.squeeze(DDS['datatypes']).ndim != 1
            or max(np.shape(DDS['datatypes'])) != nc):
            error = 'datatypes'
        else:
            DDS['datatypes'] = np.reshape(DDS['datatypes'],
                np.size(DDS['datatypes']))

        if 'depth' not in DDS or type(DDS['depth']) not in (int, float):
            error = 'depth'

        if 'fs' not in DDS or type(DDS['fs']) not in (int, float):
            error = 'fs'

        if 'data' in DDS:
            if np.ndim(DDS['data']) < 2 or np.shape(DDS['data'])[1] != nc:
                error = 'data'
        else:
            DDS['data'] = np.zeros((1, nc))

        if len(error) != 0:
            print('\nInstrument data structure error: field [{}] not specified '
                'correctly'.format(error))
            DDS = []
            return DDS

    # -------------------------------------------------------------------------
    # Special matrix
    # -------------------------------------------------------------------------
    if type_ == 2:
        if type(DDS) != dict:
            print('DIWASP data_check: Special matrix data type is not a '
                'structure')

        if 'freqs' in DDS and np.squeeze(DDS['freqs']).ndim == 1:
            nf = np.size(DDS['freqs'])
        else:
            error = 'freqs'

        if 'dirs' in DDS and np.squeeze(DDS['dirs']).ndim == 1:
            nd = np.size(DDS['dirs'])
        else:
            error = 'dirs'

        if 'S' in DDS:
            if (np.shape(DDS['S'])[0] != nf or np.ndim(DDS['S']) < 2 or
                np.shape(DDS['S'])[1] != nd):
                error = 'S'
            else:
                DDS['S'] = []

        if 'xaxisdir' in DDS:
            if type(DDS['xaxisdir']) not in (int, float):
                error = 'xaxisdir'
        else:
            DDS['xaxisdir'] = SM['xaxisdir']

        if 'dunit' not in DDS:
            DDS['dunit'] = 'cart'

        if 'funit' not in DDS:
            DDS['funit'] = 'hz'

        if len(error) != 0:
            print('\nSpectral matrix structure error: field [{}] not '
                'specified correctly'.format(error))
            DDS = []
            return DDS

    #--------------------------------------------------------------------------
    # Estimation parameters
    #--------------------------------------------------------------------------
    if type_ == 3:
        if type(DDS) != dict:
            print('DIWASP data_check: Estimation parameter data type is not a '
                'structure')

        if 'dres' in DDS:
            if type(DDS['dres']) not in (int, float):
                error = 'dres'
            elif DDS['dres'] < 10:
                DDS['dres'] = 10
                warnings.warn('dres is too small and has been set to 10')

        else:
            DDS['dres'] = EP['dres']

        if 'nfft' in DDS:
            if type(DDS['nfft']) not in (int, float):
                error = 'nfft'
            elif DDS['nfft'] < 64:
                DDS['nfft'] = 64
                warnings.warn('nfft is too small and has been set to 64')
        else:
            DDS['nfft'] = EP['nfft']

        if 'iter' in DDS:
            if type(DDS['iter']) not in (int, float):
                error = 'iter'
        else:
            DDS['iter'] = EP['iter']

        if 'smooth' in DDS:
            if DDS['smooth'].upper() != 'OFF':
                DDS['smooth'] = 'ON'
        else:
            DDS['smooth'] = 'ON'

        if 'method' in DDS:
            if DDS['method'].upper() not in ('DFTM', 'EMLM', 'IMLM', 'EMEP',
                'BDM'):
                error = 'method'
        else:
            DDS['method'] = EP['method']

        if len(error) != 0:
            print('\nEstimation parameters structure error: field [{}] not '
                'specified correctly')
            DDS = []
            return DDS

    if type_ not in (1, 2, 3):
        print()
        warnings.warn('DIWASP data_check: Data type unknown')
        DDS = []

    return DDS
