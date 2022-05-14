import pickle
import pandas as pd
import numpy as np
from tsfresh import extract_relevant_features

def extract_features(exo_data_dir, non_exo_data_dir, extracted_features_save_dir):
    '''
    Conducts feature extraction using tsfresh by first creating
    the appropriate input for tsfresh. Combines the exo data
    and non_exo data into a single dataframe, with each row
    containing a single rv measurement.
    
    Params:
        - 'exo_data_dir': name/directory of raw exoplanetary rv measurements (pickle)
        - 'non_exo_data_dir': name/directory of raw non-exoplanetary rv measurements (pickle)
        - 'extracted_features_save_dir': name/directory to save extracted features (pickle)
    '''
    
    # Load rv data
    with open(exo_data_dir, 'rb') as f:
        exo_data = pickle.load(f)
        
    with open(non_exo_data_dir, 'rb') as f:
        non_exo_data = pickle.load(f)
        
    
    # Combine rv data into table for tsfresh
    column_names = ['id', 'name', 'x']
    df = pd.DataFrame(columns = column_names)
    labels = []
    
    id_val = 0
    
    # Take each rv measurement and append to dataframe
    for item in exo_data:
        length = len(item[0])
        ids = np.full(length, id_val)

        times = np.array(item[0])
        rvs = np.array(item[1])
        to_add = np.array([ids, times, rvs]).T
        to_add = to_add[~np.isnan(to_add).any(axis=1)]  # remove any nan measurements
        if (len(to_add) == 1):  # Ignore any single rv measurements
            continue

        new_df = pd.DataFrame(to_add, columns = df.columns)
        df = pd.concat([df, new_df], ignore_index=True)
        labels.append(1)
        id_val += 1
    
    for item in non_exo_data:
        length = len(item[0])
        ids = np.full(length, id_val)

        times = np.array(item[0])
        rvs = np.array(item[1])
        to_add = np.array([ids, times, rvs]).T
        to_add = to_add[~np.isnan(to_add).any(axis=1)]
        if (len(to_add) == 1):
            continue

        new_df = pd.DataFrame(to_add, columns = df.columns)
        df = pd.concat([df, new_df], ignore_index=True)
        labels.append(1)
        id_val += 1
    
    df = df.astype(float)
    ids = np.array([i for i in range(0, id_val)])
    data = {'id':ids, 'y':labels}
    y = pd.DataFrame(data['y']).squeeze()
    
    # Conduct feature extraction
    features_filtered_direct = extract_relevant_features(df, y, 
                                                         column_id='id', column_sort='time')
    
    x_data = features_filtered_direct.to_numpy()
    y_data = y.to_numpy()
    
    rv_data_features_extracted = (x_data, y_data)
    
    with open(extracted_features_save_dir, 'wb') as f:
        pickle.dump(rv_data_features_extracted, f)

if __name__ == '__main__':
    # Feature extraction for real data:
    main = '../data/real_data/'
    extract_features(main + 'all_exo_data_nasa', main + 'all_non_exo_data', 'real_rv_data_features_extracted')
    
    # Feature extraction for simulated data:
    main = '../data/simulated_data/generated_data/'
    extract_features(main + 'simulated_exo_rvs', main + 'simulated_non_exo_rvs', 'simulated_rv_data_features_extracted')
    