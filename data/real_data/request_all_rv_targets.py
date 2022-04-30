import pickle
from dace.spectroscopy import Spectroscopy

'''
Request spectroscopic candidates with radial velocity data from DACE database.
Save requests to a file (large).
'''

def request_spectroscopic_targets():
    '''
    Query 100,000 spectroscopic targets (most will be duplicates, but this ensures we get almost all unique targets)
    and save results to pickle file. This file is accessed in request_non_exo_rv.py to remove duplicates and obtain
    the radial velocity data for non-exoplanetary systems.
    '''
    
    observedTargets = Spectroscopy.query_database(limit=100000)
    
    with open('observedTargets', 'wb') as fp:
        pickle.dump(observedTargets, fp)

if __name__ == '__main__':
    request_spectroscopic_targets()