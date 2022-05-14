'''
Retrieve radial velocity time-series data from DACE database of systems
not known to have any exoplanets.


DACE: https://dace.unige.ch/dashboard/index.html
'''

import numpy as np
from dace.spectroscopy import Spectroscopy
from dace.exoplanet import Exoplanet
import pickle
import itertools

def get_all_exoplanet_names():
    '''
    Obtain a set of all host star names that contain exoplanets.

    Returns:
        - set of all the host star names
    '''

    exoplanet_data = Exoplanet.query_database(limit = 50000)
    pl_names = exoplanet_data['obj_id_catname']
    
    all_exoplanets = set()
    for name in pl_names:
        if (name[-1].islower()):
            name = name[:-2].replace(' ', '').lower()
        else:
            name = name.replace(' ', '').lower()

        name = name.replace('-', '')
        all_exoplanets.add(name)
        
    return all_exoplanets

def get_all_non_exoplanet_names(all_exoplanets):
    '''
    Obtain a list of all host star names that do not have exoplanets, cross-referencing
    the set of exoplanetary systems so that the too classes do not overlap.

    Params:
        - 'all_exoplanets': a set of all the host star names containing exoplanets

    Returns:
        - A set of the names of all the systems not confirmed to have exoplanets.
    '''

    # Get the names of all systems with spectroscopic data
    observedTargets = None
    with open('observedTargets','rb') as f: 
        observedTargets = pickle.load(f)
    
    names = observedTargets['obj_id_catname']
    processed_names = [i.replace('-', '').lower() for i in names]
    ids = observedTargets['obj_id_daceid']
    dictionary = dict(zip(processed_names, ids))
    
    # Keep track of all the systems with IDs corresponding to exoplanets
    all_exoplanet_ids = set()
    for item in all_exoplanets:
        try:
            dace_id = dictionary[item]
            all_exoplanet_ids.add(dace_id)
        except KeyError:
            continue
    
    # Get all the unique systems that do not contain exoplanets
    non_exoplanet_names = set()
    seen_ids = set()
    for (name, dace_id) in itertools.zip_longest(names, ids):
        if dace_id in all_exoplanet_ids or dace_id in seen_ids:
            continue
        else:
            non_exoplanet_names.add(name)
            seen_ids.add(dace_id)
    
    return non_exoplanet_names

def request_rv_data(all_non_exoplanet_names):
    '''
    Obtain the radial velocity data of the non-exoplanetary systems, querying DACE's online
    Spectroscopy database. Saves the data to a pickle file.

    Params:
        - 'all_non_exoplanet_names': the set of the names of all host stars not confirmed
                                    to have exoplanets
    '''


    all_rv_non_exoplanet_data = []
    count = 0
    for name in all_non_exoplanet_names:
        count += 1
        spectro_time_series = Spectroscopy.get_timeseries(name, sorted_by_instrument=False)
        dates = spectro_time_series['rjd']
        
        if (len(dates) == 0):
            continue
        
        rvs = spectro_time_series['rv']
        data = np.stack((dates, rvs))
        all_rv_non_exoplanet_data.append(data)
        print(count)
    
    with open ("all_non_exo_data", "wb") as fp:
        pickle.dump(all_rv_non_exoplanet_data, fp)
    

if __name__ == '__main__':
    all_exoplanet_names = get_all_exoplanet_names()
    all_non_exoplanet_names = get_all_non_exoplanet_names(all_exoplanet_names)
    request_rv_data(all_non_exoplanet_names)