'''
This file contains the implementation of model tuning with
Keras' hyperband. We optimize based off of AUC.
'''

from tensorflow import keras
from tensorflow.keras import layers, models, metrics, regularizers
import keras_tuner as kt
import pickle
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def model_builder_real_fc(hp):
    '''
    Defines the method used to tune the learning rate and number of units
    in hidden layer for the fully connected neural network model for the real data.
    
    Params:
        - hp: hypermodel instance
        
    Returns:
        - Compiled model to tune
    '''
    
    hp_units = hp.Int('units', min_value=16, max_value=512, step=16)
    hp_lr = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4])
    
    X_input = layers.Input(329,)
    X = layers.Dense(hp_units, activation = 'relu', kernel_regularizer = regularizers.L2(0.01))(X_input)
    X = layers.Dense(1, activation = 'sigmoid')(X)
    
    model = models.Model(inputs=X_input, outputs=X, name='baseline')

    model.compile(optimizer = optimizers.Adam(learning_rate=hp_lr), 
                  loss = 'binary_crossentropy', metrics = ['acc', metrics.AUC()])
    return model

def model_builder_sim_fc(hp):
    '''
    Defines the method used to tune the learning rate and number of units in each
    of the hidden layers for the simulated dataset's fully-connected neural network
    model.
    
    Params:
        - hp: hypermodel instance
    
    Returns:
        - Compiled model used for tuning
    '''
    hp_units_1 = hp.Int('units1', min_value=16, max_value=512, step=16)
    hp_units_2 = hp.Int('units2', min_value=16, max_value=512, step=16)
    hp_units_3 = hp.Int('units3', min_value=16, max_value=512, step=16)
    hp_lr = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4])
    
    X_input = layers.Input(381,)
    X = layers.Dense(hp_units_1, activation = 'relu')(X_input)
    X = layers.Dense(hp_units_2, activation = 'relu')(X)
    X = layers.Dense(hp_units_3, activation = 'relu')(X)
    X = layers.Dense(1, activation = 'sigmoid')(X)
    
    model = models.Model(inputs=X_input, outputs=X, name='baseline')

    model.compile(optimizer = optimizers.Adam(learning_rate=hp_lr), 
                  loss = 'binary_crossentropy', metrics = ['acc', metrics.AUC()])
    return model



def model_builder_cnn_lstm_real(hp):
    '''
    Defines the method used to tune the learning rate and number of units in each
    hidden layer in the real dataset's CNN-LSTM model.
    
    Params:
        - hp: keras hypermodel instance
        
    Retruns:
        - Compiled model for tuning.
    '''
    hp_units_1 = hp.Int('units1', min_value=16, max_value=512, step=16)
    hp_units_2 = hp.Int('units2', min_value=16, max_value=512, step=16)
    hp_lr = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4])
    
    X_input = layers.Input((pad_amount, 1))
    X = layers.Conv1D(128, 3, activation = 'relu')(X_input)
    X = layers.Conv1D(128, 3, activation = 'relu')(X)
    X = layers.MaxPooling1D()(X)
    X = layers.Conv1D(64, 3, activation = 'relu')(X)
    X = layers.Conv1D(64, 3, activation = 'relu')(X)
    #X = layers.MaxPooling1D()(X)
    X = layers.Bidirectional(layers.LSTM(64))(X)
    X = layers.Dense(hp_units_1, activation = 'relu')(X)
    X = layers.Dense(hp_units_2, activation = 'relu')(X)
    X = layers.Dense(1, activation = 'sigmoid')(X)
    
    
    model = models.Model(inputs = X_input, outputs = X, name='real_lstm')
    model.compile(optimizer = optimizers.Adam(learning_rate=hp_lr), 
                  loss = 'binary_crossentropy', metrics = ['acc', metrics.AUC()])
    
    return model

def model_builder_cnn_lstm_sim(hp):
    '''
    Defines the method used to tune the learning rate and number of units in each
    of the hidden layers in the simualated dataset's CNN-LSTM model.
    
    Params:
        - hp: hypermodel instance
    
    Returns:
        - Compiled model used for tuning
    '''
    
    hp_units_1 = hp.Int('units1', min_value=16, max_value=512, step=16)
    hp_units_2 = hp.Int('units2', min_value=16, max_value=512, step=16)
    hp_lr = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4])
    
    X_input = layers.Input((pad_amount, 1))
    X = layers.Conv1D(128, 3, activation = 'relu')(X_input)
    X = layers.Conv1D(128, 3, activation = 'relu')(X)
    X = layers.MaxPooling1D()(X)
    X = layers.Bidirectional(layers.LSTM(64))(X)
    X = layers.Dense(hp_units_1, activation = 'relu')(X)
    X = layers.Dense(hp_units_2, activation = 'relu')(X)
    X = layers.Dense(1, activation = 'sigmoid')(X)
    model = models.Model(inputs = X_input, outputs = X, name='sim_lstm')
    
    model.compile(optimizer = optimizers.Adam(learning_rate=hp_lr), 
                  loss = 'binary_crossentropy', metrics = ['acc', metrics.AUC()])
    return model
    


if __name__ == '__main__':
    # Sample tuner search for real data, fc network, others are done similarly
    with open('../feature_extration/real_rv_data_features_extracted', 'rb') as f:
        x_data, y_data = pickle.load(f)
    
    # Preprocess the data as done normally before searching
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=0)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=(0.2) / (1-0.2), random_state=0)
    normalizer = preprocessing.StandardScaler()
    normalized_train_X = normalizer.fit_transform(x_train)
    normalized_val_x = normalizer.transform(x_val)
    normalized_test_x = normalizer.transform(x_test)
    
    # Conduct hyperparameter searching
    tuner = kt.Hyperband(model_builder_real_fc,
                    objective = kt.Objective("val_auc", direction="max"),
                    max_epochs = 20)
    
    tuner.search(normalized_train_X, y_train, epochs=50, validation_data=(normalized_val_x, y_val))
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
    print(best_hps)