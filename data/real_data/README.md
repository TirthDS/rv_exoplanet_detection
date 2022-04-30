## Obtaining Real Radial Velocity Data
### Host Stars With Exoplanets
For host stars containing exoplanetary systems, raw radial velocity data was retreived from NASA's Exoplanet Archive's constributed datasets page for radial velocities [here](https://exoplanetarchive.ipac.caltech.edu/bulk_data_download/#TSD).
There is a total of 1070 radial velocity curves of host stars containing exoplanets.
- Script to Parse Through Data: [obtain_exo_rv](obtain_exo_rv.py)
  - Data is not uploaded to this repository and is available at NASA's Exoplanet Archive.
- Raw Extracted Radial Velocities (Pickle): [all_exo_data_nasa](all_exo_data_nasa)

### Host Stars Without Exoplanets
For host stars not containing exoplanetary systems, raw radial velocity data was retreived from DACE's database [here](https://dace.unige.ch/observationSearch/?observationType=[%22spectroscopy%22]). There is a total of 4597 radial velocity curves of host stars not known to contain exoplanets. However, only 4270 radial velocity curves have more than 1 datapoint, necessary for feature extraction.
- Script to Query All RV Targets (exoplanetary and non-exoplanetary systems): [request_all_rv_targets.py](request_all_rv_targets.py)
  - Requested data is not uploaded to this repository.
- Script to Obtain Non-Exoplanetary Systems and Query Radial Velocity Data: [request_non_exo.py](request_non_exo_rv.py)
- Raw Extracted Radial Velocities for Non-Exoplanetary Systems (Pickle): [all_non_exo_data](all_non_exo_data)
