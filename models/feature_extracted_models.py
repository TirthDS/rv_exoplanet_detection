from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tensorflow import keras
from tensorflow.keras import layers, models, metrics, optimizers
import pickle

def simulated_base_model(feature_length):
    '''
    Defines the baseline fully-connected neural network model
    for the simulated data. Number of units in each hidden layer 
    was optimized using Keras hyperband.
    
    Params:
        - 'feature_length': the number of features for this dataset
        
    Returns:
        - The model being trained on.
    '''
    
    X_input = layers.Input(feature_length,)
    X = layers.Dense(208, activation='relu')(X_input)
    X = layers.Dense(160, activation='relu')(X)
    X = layers.Dense(512, activation = 'relu')(X)
    X = layers.Dense(1, activation='sigmoid')(X)
    
    model = models.Model(inputs = X_input, outputs=X, name='baseline')
    
    return model

def real_base_model(feature_length):
    '''
    Defines the baseline fully-connected neural network model
    for the real rv data. Number of units in hidden layer was 
    optimized using Keras' tensor band.
    
    Params:
        - 'feature_length': the number of features for this dataset (will be different
                            than the simulated data)
    
    Returns:
        - The model being trained on.
    '''

    X_input = layers.Input(feature_length,)
    X = layers.Dense(64, activation='relu', kernel_regularizer = regularizers.L2(0.01))(X_input)
    X = layers.Dense(1, activation='sigmoid')(X)
    model = models.Model(inputs = X_input, outputs=X, name='baseline_real')
    
    return model

def preprocess(x_data, y_data, dev_test_split):
    '''
    Splits data into train/dev/test sets and normalizes the data feature-wise.
    
    Params:
        - 'x_data': the extracted features for each system
        - 'y_data': the labels (0 for non-exoplanet, 1 for exoplanet)
        - 'dev-test-split': the percentage of data to use for dev and test sets
    
    Returns:
        - The normalized training, validation, and test datasets.
    '''
    
    # Split
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=dev_test_split, random_state=0)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=(dev_test_split) / (1-dev_test_split), random_state=0)
    
    # Normalize
    normalizer = preprocessing.StandardScaler()
    normalized_train_x = normalizer.fit_transform(x_train)
    normalized_val_x = normalizer.transform(x_val)
    normalized_test_x = normalizer.transform(x_test)
    
    return (normalized_train_x, y_train), (normalized_val_x, y_val), (normalized_test_x, y_test)

def fit(model, train_data, val_data, epochs, batch_size, learning_rate, threshold):
    '''
    Splits data into train/dev/test sets based off of 'dev_test_split' and normalizes
    based off of the training set. Fits the appropriate model, evaluates on the test
    set and returns test loss, accuracy, precision, and recall.
    
    Params:
        - 'model': the model being used for fitting
        - 'train_data': tuple containing training data (x_train, y_train)
        - 'val_data': tuple containing validation data (x_val, y_val)
        - 'epochs': the number of iterations to train for
        - 'batch_size': the size of batches to use for batch gradient descent 
        - 'learning_rate': the learning rate to use for gradient descent
        - 'threshold': the decision threshold to use for recall/precision metrics
    
    Returns:
        - Testing data/labels
        - Validation data/labels
        - History object containing a record of training loss values and metrics
    '''
    # Use binary crossentropy loss with adam optimizer
    model.compile(optimizer = optimizers.Adam(learning_rate=learning_rate), loss = 'binary_crossentropy', 
                  metrics = ['accuracy', metrics.Recall(thresholds=threshold), metrics.Precision(thresholds=threshold),
                             metrics.AUC()])
    
    # Run Model
    history = model.fit(train_data[0], train_data[1], validation_data = val_data, epochs = epochs, 
                        batch_size=batch_size, shuffle=True)
    
    return history

def evaluate(model, test_data):
    '''
    Evaluates the model on the unseen testing dataset and reports the AUC,
    recall, precision, and F1 score.
    
    Params:
        - 'model': the model that was fitted on the training data
        - 'test_data': a tuple containing the test data (x_test, y_test)
        
    Returns:
        - Metrics used: loss, accuracy, recall, precision, and auc (subject to
          the thresholds chosen in fitting)
    '''
    loss, acc, recall, precision, auc = model.evaluate(test_data[0], test_data[1])
    print("AUC: " + str(auc))
    print("Recall: " + str(recall))
    print("Precision: "+ str(precision))
    print("F1 Score: " + str(2 * (recall * precision) / (recall + precision)))
    return loss, acc, recall, precision, auc


if __name__ == '__main__':
    # Run model on real rv data
    file = '../feature_extraction/real_rv_data_features_extracted'
    with open(file, 'rb') as f:
        x_data_real, y_data_real = pickle.load(f)
    
    model = real_base_model(x_data_real.shape[1])
    train, val, test = preprocess(x_data_real, y_data_real, 0.2)
    history = model.fit(model, train, val, 15, 32, 0.001, 0.2)
    print('Real Data Model: ')
    loss, acc, recall, precision, auc = evaluate(model, test)
    
    # Run model on simulated rv data
    file = '../feature_extraction/simulated_rv_data_features_extracted'
    with open(file, 'rb') as f:
        x_data_sim, y_data_sim = pickle.load(f)
    
    model = simulated_base_model(x_data_sim.shape[1])
    train, val, test = preprocess(x_data_sim, y_data_sim, 0.2)
    history = model.fit(model, train, val, 10, 32, 0.001, 0.2)
    print('Simulated Data Model: ')
    loss, acc, recall, precision, auc = evaluate(model, test)