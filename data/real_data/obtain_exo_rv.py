'''
Parse through NASA's exoplanet archive contributed RV dataset
for radial velocity time-series data of exoplanetary systems.
Contents are in 'dataset' directory.
'''
import numpy as np
import glob
import re
import pickle

def parse_files(all_files):
    '''
    Parses through the directory containing all of NASA's contributed
    radial velocity curves for exoplanetary systems and saves the data.

    Params:
        - 'all_files': a list containing all of the files storing the RV data

    Returns:
        - a list of all the radial velocity data (times, rvs), a list of each star's date units,
          and a list of each star's rv measurement units (to check for unit consistency)
    '''

    all_exo_data_nasa = []
    date_units = []
    rv_units = []

    # Cannot use pandas due to slight variations in data file formats
    for file in all_files:
        file1 = open(file, 'r')
        lines = file1.readlines()
        
        # Search for the index of the beginning of RVs
        beginning_index = [idx for idx, s in enumerate(lines) if '|' in s][-1] + 1

        # Get index of the date units:
        date_id_index = [idx for idx, s in enumerate(lines) if 'DATE_UNITS' in s][0]

        # Get index of the rv units:
        rv_id_index = [idx for idx, s in enumerate(lines) if 'VALUE_UNITS' in s][0]

        # Get the date and rv unit types:
        char = "\""
        date_type_idx = [i.start() for i in re.finditer(char, lines[date_id_index])]
        if (len(date_type_idx) == 0):
            char = "\'"
            date_type_idx = [i.start() for i in re.finditer(char, lines[date_id_index])]
        
        rv_type_idx = [i.start() for i in re.finditer(char, lines[rv_id_index])]

        date_type = lines[date_id_index][date_type_idx[0]+1:date_type_idx[1]].lower().replace(' ', '')
        rv_type = lines[rv_id_index][rv_type_idx[0]+1:rv_type_idx[1]].lower().replace(' ', '')
        
        # Grab rvs and remove all 'nan' values
        rv = lines[beginning_index:]
        rvs = np.array([item.split() for item in rv]).astype(float)
        rvs = rvs[~np.isnan(rvs).any(axis=1)].T

        dates = rvs[0]
        rvs = rvs[1]

        if (len(dates) == 0):
            continue

        date_units.append(date_type)
        rv_units.append(rv_type)

        data = np.stack((dates, rvs))
        all_exo_data_nasa.append(data)
        
    return all_exo_data_nasa, date_units, rv_units

if __name__ == '__main__':
    # Assume all files are extracted to this relative directory
    all_files = glob.glob('nasa_rv_dataset/*')
    all_exo_data_nasa, date_units, rv_units = parse_files(all_files)
    
    # Ensure consistency of units for rv measurements
    assert(np.all([item=='m/s' for item in rv_units]))
    assert(np.all([item=='days' for item in date_units]))
    
    # Save the extracted data
    with open('all_exo_data_nasa', 'wb') as f:
        pickle.dump(all_exo_data_nasa, f)