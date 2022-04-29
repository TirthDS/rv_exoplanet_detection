## Obtaining Real Radial Velocity Data
### Host Stars With Exoplanets
For host stars containing exoplanetary systems, raw radial velocity data was retreived from NASA's Exoplanet Archive's constributed datasets page for radial velocities [here](https://exoplanetarchive.ipac.caltech.edu/bulk_data_download/#TSD).
There are a total of 1070 radial velocity curves of host stars containing exoplanets.
- Script to Parse Through Data: [obtain_exo_rv.py](https://github.com/TirthDS/rv_exoplanet_detection/blob/main/data/real_data/obtain_exo_rv.p)
- Raw Extracted Radial Velocities (Pickle): [all_exo_data_nasa](https://github.com/TirthDS/rv_exoplanet_detection/blob/main/data/real_data/all_exo_data_nasa)

### Host Stars Without Exoplanets
For host stars not containing exoplanetary systems, raw radial velocity data was retreived from DACE's database [here](https://dace.unige.ch/observationSearch/?observationType=[%22spectroscopy%22]). There's a total of
__ radial velocity curves of host stars not known to contain exoplanets.
- Script to Query All RV Targets (exoplanetary and non-exoplanetary systems): 
- Script to Obtain Non-Exoplanetary Systems and Query Radial Velocity Data: [request_non_exo.py](https://github.com/TirthDS/rv_exoplanet_detection/blob/main/data/real_data/all_exo_data_nasa)
- Raw Extracted Radial Velocities for Non-Exoplanetary Systems (Pickle): [all_non_exo_data](https://github.com/TirthDS/rv_exoplanet_detection/blob/main/data/real_data/all_non_exo_data)
