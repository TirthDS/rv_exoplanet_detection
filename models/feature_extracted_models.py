from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tensorflow import keras
from tensorflow.keras import layers, models, metrics
import pickle

def simulated_base_model(feature_length):
    '''
    Defines the baseline fully-connected neural network model
    for the simulated data.
    
    Params:
        - 'feature_length': the number of features for this dataset
        
    Returns:
        - The model being trained on.
    '''
    
    X_input = layers.Input(feature_length,)
    X = layers.Dense(64, activation='relu')(X_input)
    X = layers.Dense(1, activation='sigmoid')(X)
    model = models.Model(inputs = X_input, outputs=X, name='baseline_simulated')
    
    return model

def real_base_model(feature_length):
    '''
    Defines the baseline fully-connected neural network model
    for the real rv data.
    
    Params:
        - 'feature_length': the number of features for this dataset (will be different
                            than the simulated data)
    
    Returns:
        - The model being trained on.
    '''

    X_input = layers.Input(feature_length,)
    X = layers.Dense(32, activation='relu')(X_input)
    X = layers.Dense(1, activation='sigmoid')(X)
    model = models.Model(inputs = X_input, outputs=X, name='baseline_real')
    
    return model

def fit_and_analysis(model, x_data, y_data, dev_test_split, epochs, batch_size):
    '''
    Splits data into train/dev/test sets based off of 'dev_test_split' and normalizes
    based off of the training set. Fits the appropriate model, evaluates on the test
    set and returns test loss, accuracy, precision, and recall.
    
    Params:
        - 'model': the model being used for fitting
        - 'x_data': the extracted features for each system
        - 'y_data': the labels (0 for non-exoplanet, 1 for exoplanets
        - 'dev_test_split': the percentage of data to use for dev and test sets
        - 'epochs': the number of iterations to train for
        - 'batch_size': the size of batches to use for batch gradient descent 
    
    Returns:
        - An array of test loss, accuracy, precision, and recall.
    '''
    # Split data and normalize feature-wise
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=dev_test_split, random_state=0)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=(dev_test_split) / (1-dev_test_split), random_state=0)
    
    # Normalize
    normalizer = preprocessing.StandardScaler()
    normalized_train_X = normalizer.fit_transform(x_train)
    normalized_val_x = normalizer.transform(x_val)
    normalized_test_x = normalizer.transform(x_test)
    
    # Use binary crossentropy loss with adam optimizer
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', 
                  metrics = ['accuracy', metrics.Precision(), metrics.Recall()])
    
    # Run Model
    history = model.fit(normalized_train_X, y_train, validation_data = (normalized_val_x, y_val), epochs = epochs, 
                        batch_size=batch_size, shuffle=True)
    
    # Evaluate on test set
    return model.evaluate(normalized_test_x, y_test)


if __name__ == '__main__':
    # Run model on real rv data
    file = '../feature_extraction/real_rv_data_features_extracted'
    with open(file, 'rb') as f:
        x_data_real, y_data_real = pickle.load(f)
    
    model = real_base_model(x_data_real.shape[1])
    real_results = fit_and_analysis(model, x_data_real, y_data_real, 0.2, 15, 32)
    print("Real RV Test Results: " + str(real_results))
    
    # Run model on simulated rv data
    file = '../feature_extraction/simulated_rv_data_features_extracted'
    with open(file, 'rb') as f:
        x_data_sim, y_data_sim = pickle.load(f)
    
    model = simulated_base_model(x_data_sim.shape[1])
    sim_results = fit_and_analysis(model, x_data_sim, y_data_sim, 0.1, 5, 32)
    print("Simulated RV Test Results: " + str(sim_results))