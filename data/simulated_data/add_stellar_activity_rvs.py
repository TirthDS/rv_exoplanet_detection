'''
Adds the stellar activity radial velocities to the simulated exoplanetary
systems to produce 10,000 rv curves of exoplanetary systems and 10,000 
rv curves of nonexoplenatary systems with stellar activity rvs taken
into account.
'''
import numpy as np
import pickle
import random
import glob

        
def exo_stellar_activity_rvs():
    '''
    Uses 10,000 of the randomly generated stellar activity radial velocity curves
    from SOAP 2.0 and adds them to the 10,000 exoplanet radial velocity curves. Applies
    a Guassian error of 5 m/s to simulate instrumental error and other random stellar
    activity.
    
    This function assumes that stellar rotation periods used for SOAP 2.0
    have been saved ('stellar_rot_periods') and solar rv curves have been fully
    generated from SOAP 2.0. This function also assumes 10,000  raw exoplanetary 
    rv curves have already been generated ('simulated_rvs_no_solar'). See
    'soap2_config_generation.py'.
    
    Returns: 
       - List of files containing solar rvs injected into the exoplanetary rv
         systems.
    '''
    stellar_rv_files = glob.glob('generated_data/solar_rvs/outputs/*') # Assume all solar rv curves in relative directory
    
    with open('generated_data/simulated_rvs_no_solar', 'rb') as f: # Load file containing raw simulated rvs
        exo_rvs = pickle.load(f)

    with open('generated_data/stellar_rot_periods', 'rb') as f: # Load file containing saved stellar rotation periods used for SOAP 2.0
        stellar_rots = pickle.load(f)
    
    stellar_rv_files_added_to_exoplanets = []
    simulated_rvs_exo = []

    # Inject planetary system rvs into stellar rv
    for i in range(len(stellar_rots)):
        file = [match for match in stellar_rv_files if str(stellar_rots[i]) in match][0]

        stellar_rv_files_added_to_exoplanets.append(file)
        f = open(file, 'r')
        lines = f.readlines()
        rvs = lines[2:]
        f.close()

        data = np.array([item.split() for item in rvs]).astype(float)
        stellar_activity_rvs = data.T[2]
        exoplanet_rvs = exo_rvs[i][1]
        new_rv = stellar_activity_rvs + exoplanet_rvs + (5 * np.random.randn(len(exoplanet_rvs)))
        dates = exo_rvs[i][0]

        data = np.stack((dates, new_rv))
        simulated_rvs_exo.append(data)
    
    # Save simulated rvs
    with open('generated_data/simulated_exo_rvs', 'wb') as f:
        pickle.dump(simulated_rvs_exo, f)
    
    return stellar_rv_files_added_to_exoplanets

def non_exo_stellar_activity_rvs(stellar_rv_files_seen):
    '''
    Parses through the remaining solar activity rv files and adds instrumental
    noise. The saved radial velocity data represents the 10,000 simulated
    systems without exoplanets.
    
    Params:
       - stellar_rv_files_seen: list of files outputted by 'exo_stellar_activity_rvs'
    '''
    
    simulated_rvs_no_exo = []

    # Retrieve solar rv signals and add Gaussian noise 5 m/s
    for file in stellar_rv_files:
        if file not in stellar_rv_files_added_to_exoplanets:
            f = open(file, 'r')
            lines = f.readlines()
            rvs = lines[2:]
            f.close()

            data = np.array([item.split() for item in rvs]).astype(float)
            stellar_activity_rvs = data.T[2] + (5 * np.random.randn(len(data.T[2])))
            phases = data.T[0]

            # Convert phases to back to date times
            rotation_period = float(file[27:-4])
            max_num_periods_to_extend_to = np.random.randint(2, 50)
            n = np.random.randint(1, max_num_periods_to_extend_to, size = len(phases)) 
            dates = (n * rotation_period) + (phases * rotation_period)
            data = np.stack((dates, stellar_activity_rvs))

            # Remove random date/rv indices (make it more realistic)
            num_points_to_remove = np.random.randint(0, len(dates) - 10)
            indices_to_remove = random.sample(range(len(dates)), num_points_to_remove)
            data = np.delete(data, indices_to_remove, axis = 1)

            simulated_rvs_no_exo.append(data)
       
    # Save simulated rvs
    with open('generated_data/simulated_non_exo_rvs', 'wb') as f:
        pickle.dump(simulated_rvs_no_exo, f)

if __name__ == '__main__':
    seen_files = exo_stellar_activity_rvs()
    non_exo_stellar_activity_rvs(seen_files)