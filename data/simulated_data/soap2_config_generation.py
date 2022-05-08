'''
These sets of functions prepare/generate files for usage in SOAP 2.0 to generate
stellar activity radial velocity curves.

1. 10,000 rv curves for exoplanetary systems are simulated
2. Based off of the generated dates, phase files are constructed,
   indicating phases to generate rv values at.
3. Stellar configuration files for input into SOAP 2.0 are generated.
4. Phase files and stellar configuration files are fed into SOAP 2.0's
   source code: http://www.astro.up.pt/resources/soap2/. Solar rv
   data is generated: 'solar_rvs.zip'.
'''

import pickle
import numpy as np
import radvel
import glob
import shutil
import configparser


def simulate_exoplanet_rvs():
    '''
    Randomly generate 10,000 radial velocity curves and add 1 m/s Gaussian noise
    to simulate future radial velocity precision. Saves the radial velocity data
    to a pickle file: 'simulated_rvs_no_solar'.
    '''
    inst_err = 1 # m/s
    
    # Define ranges of parameter values
    num_measurements = (10, 100) # Number of rv-measurements to simulate
    orbital_frac = (0.5, 3) # Fraction of orbital period to estimate within
    periods = (1e-1, 1e4) # Orbital periods (days)
    ecc = (0, 1) # Eccentricity of orbit
    k = (1e-1, 1e3) # RV semi-amplitude (m/s)
    w = (0, 2*np.pi) # Argument of Periastron
    
    num_pl = 10000
    simulated_rvs_no_solar_activity = []

    # Monte-carlo process to generate radial velocity curves
    for i in range(num_pl):

        # Randomly sample parameters
        params = radvel.Parameters(1, basis = 'per tc e w k')
        pl_per = np.exp(np.random.uniform(np.log(periods[0]), np.log(periods[1])))
        params['per1'] = radvel.Parameter(value = pl_per)
        pl_ecc = np.random.uniform(ecc[0], ecc[1])
        params['e1'] = radvel.Parameter(value = pl_ecc)
        pl_w = np.random.uniform(w[0], w[1])
        params['w1'] = radvel.Parameter(value = pl_w)
        pl_k = np.exp(np.random.uniform(np.log(k[0]), np.log(k[1])))
        params['k1'] = radvel.Parameter(value = pl_k)

        params['dvdt'] = radvel.Parameter(value=0)
        params['curv'] = radvel.Parameter(value=0)

        # Get range of times with simulated period
        pl_measurements = np.random.randint(num_measurements[0], num_measurements[1])
        pl_orbital_frac = np.random.uniform(orbital_frac[0], orbital_frac[1])
        x_rv = np.sort(np.random.uniform(0, pl_per * pl_orbital_frac, size = pl_measurements))

        # Get time of periastron
        pl_tc = np.random.uniform(0, pl_per * pl_orbital_frac)
        params['tc1'] = radvel.Parameter(value = pl_tc)

        synth_model = radvel.RVModel(params=params)
        y_rv = synth_model(x_rv)
        y_rv += inst_err * np.random.randn(len(y_rv))  # Add instrumental noise, Gaussian distributed

        data = np.stack((x_rv, y_rv))
        simulated_rvs_no_solar_activity.append(data)
    
    # Save raw radial velocity measurements (no stellar activity)
    with open('generated_data/simulated_rvs_no_solar', 'wb') as f:
        pickle.dump(simulated_rvs_no_solar_activity, f)

def generate_phase_files():
    '''
    Generates phase files for the 10,000 simulated exoplanetary systems above and
    saves them to local directory. The outputs have been saved to 'phases.zip'.
    '''
    
    rotation_periods = (4, 30) # days
    stellar_rot_periods = np.empty(10000)
    for i in range(10000):
        stellar_rot_periods[i] = np.random.uniform(rotation_periods[0], rotation_periods[1])
    
    # Save stellar rotation periods (for use later)
    with open('generated_data/stellar_rot_periods', 'wb') as f:
        pickle.dump(stellar_rot_periods, f)
        
    with open('generated_data/simulated_rvs_no_solar', 'rb') as f:
        simulated_rvs_no_solar_activity = pickle.load(f)
    
    # Generate phase files:
    for i in range(len(simulated_rvs_no_solar_activity)):
        stellar_rot_period = stellar_rot_periods[i]
        dates = simulated_rvs_no_solar_activity[i][0]
        phases = (dates % stellar_rot_period) / stellar_rot_period
        np.savetxt('generated_data/phases/' + str(stellar_rot_period) + '.txt', phases)
    

def generate_config_files():
    '''
    Copies sample config file 20,000 times. Updates each file with randomly generated
    stellar parameters. 10,000 of these configuration files will reference the phase
    files generated previously. These generated files are then fed into SOAP 2.0.
    The outputs have been saved to 'configs.zip'.
    '''
    
    # Create copy of sample config file
    for i in range(20000):
        shutil.copyfile('sample_config.cfg', 'generated_data/configs/config' + str(i) + '.cfg')
        
    # Get list of all phase and config files
    phase_files = glob.glob('phases/*')
    config_files = np.sort(glob.glob('configs/*'))
    
    # Monte Carlo generation of stellar configuration files
    radius_range = (0.7, 1.4) # Radius of FGK stars
    i_range = (0, 90) # Inclination
    t_star_range = (4000, 6500) # Effective temperature, K
    t_diff_spot_range = (500, 2000) # Temperature difference between spot and effective temp
    num_active_regions_range = (1, 4) # Number of active regions
    long_range = (-180, 180) # Longitude
    lat_range = (-90, 90) # Latitude
    size_range = (0.01, 0.2) # Size of region (stellar radius)
    rotation_periods = (4, 30) # Rotational period (days)
    
    for i in range(num_simulated):
        config = configparser.ConfigParser()
        config.read(config_files[i])

        radius = np.random.uniform(radius_range[0], radius_range[1])
        i_val = np.random.uniform(i_range[0], i_range[1])
        t_star = np.random.randint(t_star_range[0], t_star_range[1])
        t_diff_spot = np.random.randint(t_diff_spot_range[0], t_diff_spot_range[1])
        prot = np.random.uniform(rotation_periods[0], rotation_periods[1])

        if (i <= 9999):
            prot = phase_files[i][7:-4]

        config.set('star', 'radius', str(radius))
        config.set('star', 'prot', str(prot))
        config.set('star', 'I', str(i_val))
        config.set('star', 'Tstar', str(t_star))
        config.set('star', 'Tdiff_spot', str(t_diff_spot))

        num_active_regions = np.random.randint(num_active_regions_range[0], num_active_regions_range[1]+1)
        for j in range(1, num_active_regions+1):
            config.set('active_regions', 'check' + str(j), '1') # Turn on active region
            reg_type = np.random.randint(0, 2)
            config.set('active_regions', 'act_reg_type' + str(j), str(reg_type))
            long = np.random.uniform(long_range[0], long_range[1])
            config.set('active_regions', 'long' + str(j), str(long))
            lat = np.random.uniform(lat_range[0], lat_range[1])
            config.set('active_regions', 'lat' + str(j), str(lat))
            size = np.random.uniform(size_range[0], size_range[1])
            config.set('active_regions', 'size' + str(j), str(size))

        if (i <= 9999): # Use phase files only for the 10,000 exoplanetary rvs
            config.set('output', 'ph_in', phase_files[i])

        with open(config_files[i], 'w') as configfile:
            config.write(configfile)

if __name__ == '__main__':
    simulate_exoplanet_rvs()
    generate_phase_files()
    generate_config_files()