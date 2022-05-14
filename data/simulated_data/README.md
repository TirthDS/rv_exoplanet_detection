# Simulated RV Data

Radial velocity measurements for 10,000 exoplanetary systems was simulated using Monte Carlo techniques with RadVel. To simulate the noise produced by stellar activity and dark spots, [SOAP 2.0](http://www.astro.up.pt/resources/soap2/) was used with randomly generated stellar configurations. A total of 20,000 stellar activity radial velocity curves were obtained. 10,000 of these stellar activity radial velocity curves were injected with the 10,000 exoplanetary radial velocity curves. A Gaussian instrumental error of 1 m/s was added to the simulated data to mimic future precision in radial velocity measurements.

### Data Summary:
- 10,000 radial velocity curves of pure stellar activity without any exoplanets
- 10,000 radial velocity curves of exoplanetary systems with solar activity

### Files:
- Simulation of 10,000 radial velocity curves without stellar activity, 10,000 phase files, and 20,000 configuration files for input into SOAP 2.0: [soap2_config_generation.py](soap2_config_generation.py)
    - 10,000 Raw Simulated Exoplanetary System RVs (Pickle): [simulated_rvs_no_solar](generated_data/simulated_rvs_no_solar)
    - 10,000 Phase Files (for each 10,000 raw simulated systems, ZIP): [phases.zip](generated_data/phases.zip)
    - 20,000 Config Generated Config Files (for all 20,000 simulations, ZIP): [configs.zip](generated_data/phases.zip)
    - Raw Simulated Stellar Activity Radial Velocities (for all 20,000 simulations, ZIP): [solar_rvs.zip](generated_data/solar_rvs.zip)

- Combining the raw 10,000 radial velocity curves with 10,000 stellar activity radial velocity curves: [add_stellar_activity_rvs.py](add_stellar_activity_rvs.py)
    - 10,000 Simulated Exoplanetary System RVs Injected with Stellar Activity RVs (Pickle): [simulated_exo_rvs](generated_data/simulated_exo_rvs)
    - 10,000 Simulated non-Exoplanetary System RVs (Pickle): [simulated_non_exo_rvs](generated_data/simulated_non_exo_rvs)
