'''
This file defines the architectures for the CNN-LSTM neural networks
on the real and simulated dataset and provides the framework to preprocess,
train, and evaluate the models on both datasets.
'''

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tensorflow import keras
from tensorflow.keras import layers, models, metrics, optimizers
import pickle

def preprocess(rv_data, max_length = 100):
    '''
    This function preprocesses the raw time series sequential data.
    
    Params:
        - rv_data: the real/simulated radial velocity data, list of (time, rv) numpy
        arrays
        - max_length: the amount of datapoints to keep/truncate/pad to.
        
    Returns:
        - A numpy array of shape (len(x_data), max_length) containing zero-padded/truncated
        radial velocity sequential data.
    '''
    
    sequence_data = np.zeros((len(x_data), max_length))

    for item in rv_data:
        item = np.array(item).astype(float)

        # Remove datapoints with 'nan' values
        sorted_sequence = item[1][np.argsort(item[0])]
        if (np.any(np.isnan(sorted_sequence))):
            ind = np.argwhere(np.isnan(sorted_sequence))
            sorted_sequence = np.delete(sorted_sequence, ind)

        # Normalize sequence before zero-padding
        norm = np.linalg.norm(sorted_sequence)
        sorted_sequence = sorted_sequence / norm

        if (len(item[0]) > max_length):
            sequence_data.append(sorted_sequence[0:max_length])
        elif (len(item[0]) < max_length):
            sequence_data.append(np.pad(sorted_sequence, (0, max_length - len(sorted_sequence))))
        else:
            sequence_data.append(sorted_sequence)
    
    return sequence_data

def fit(model, train_data, val_data, epochs, batch_size, learning_rate, threshold):
    '''
    Fits the appropriate model on the normalized simulated/real dataset.
    
    Params:
        - 'model': the model being used for fitting
        - 'train_data': tuple containing training data (x_train, y_train)
        - 'val_data': tuple containing validation data (x_val, y_val)
        - 'epochs': the number of iterations to train for
        - 'batch_size': the size of batches to use for batch gradient descent 
        - 'learning_rate': the learning rate to use for gradient descent
        - 'threshold': the decision threshold to use for recall/precision metrics
    
    Returns:
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

def simulated_cnn_lstm_model():
    '''
    Defines the model architecture for the CNN-LSTM model for the simulated
    dataset.
    
    Returns:
        - The Keras model object to be trained on.
    '''
    
    X_input = layers.Input((pad_amount, 1))
    X = layers.Conv1D(128, 3, activation = 'relu')(X_input)
    X = layers.Conv1D(128, 3, activation = 'relu')(X)
    X = layers.MaxPooling1D()(X)
    X = layers.Bidirectional(layers.LSTM(64))(X)
    X = layers.Dense(128, activation = 'relu')(X)
    X = layers.Dense(144, activation = 'relu')(X)
    X = layers.Dense(1, activation = 'sigmoid')(X)
    
    model = models.Model(inputs = X_input, outputs = X, name='sim_lstm')
    return model


def real_cnn_lstm_model():
    '''
    Defines the model architecture for the CNN-LSTM model for the real
    dataset.
    
    Returns:
        - The Keras model object to be trained on.
    '''
    X_input = layers.Input((pad_amount, 1))
    X = layers.Conv1D(128, 3, activation = 'relu')(X_input)
    X = layers.Conv1D(128, 3, activation = 'relu')(X)
    X = layers.MaxPooling1D()(X)
    X = layers.Conv1D(64, 3, activation = 'relu')(X)
    X = layers.Conv1D(64, 3, activation = 'relu')(X)
    X = layers.MaxPooling1D()(X)
    X = layers.Bidirectional(layers.LSTM(64))(X)
    X = layers.Dense(448, activation = 'relu')(X)
    X = layers.Dense(256, activation = 'relu')(X)
    X = layers.Dense(1, activation = 'sigmoid')(X)
    
    model = models.Model(inputs = X_input, outputs = X, name='real_lstm')
    return model

def load_data(exo_file, non_exo_file, max_length = 100, split = 0.2):
    '''
    Loads the exoplanetary and non-exoplanetary rv data, preprocesses them,
    and splits them into the train/dev/test sets.
    
    Params:
        - 'exo_file': path to file containing the rv data for exoplanetary systems
        - 'non_exo_file': path to file containing the rv data for non-exoplanetary systems
        - 'max_length': the maximum number of data points to pad/truncate to
        - 'split': the train/dev/test split to use (1-split/split/split)
    '''
    with open(exo_file, 'rb') as f:
        exo_data = pickle.load(f)
        
    with open(non_exo_file, 'rb') as f:
        non_exo_data = pickle.load(f)
    
    x_data = exo_data + non_exo_data
    sequence_data = preprocess(x_data, max_length = max_length)
    y_data = np.concatenate((np.ones(len(exo_data)), np.zeros(len(non_exo_data))))
    
    x_train, x_test, y_train, y_test = train_test_split(sequence_data, y_data, test_size = split, random_state=0)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = (split / (1 - split)), random_state=0)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
                             
if __name__ == '__main__':
    # Run model on real rv data
    file1 = '../data/real_data/all_exo_data_nasa'
    file2 = '../data/real_data/all_non_exo_data'
    train_real, val_real, test_real = load_data(file1, file2)
    
    model_real = real_cnn_lstm_model()
    history_real = fit(model_real, train_real, val_real, 15, 32, 0.001, 0.3)
    model_real.save('real_lstm_model')
    print('Real Data Model: ')
    loss_real, acc_real, recall_real, precision_real, auc_real = evaluate(model_real, test_real)
    
    # Run model on simulated rv data
    file1 = '../data/simulated_data/generated_data/simulated_exo_rvs'
    file2 = '../data/simulated_data/generated_data/simulated_non_exo_rvs'
    train_sim, val_sim, test_sim = load_data(file1, file2)
    
    model_sim = simulated_cnn_lstm_model()
    history_sim = fit(model_sim, train_sim, val_sim, 10, 32, 0.001, 0.26)
    model_sim.save('sim_lstm_model')
    print('Simulated Data Model: ')
    loss_sim, acc_sim, recall_sim, precision_sim, auc_sim = evaluate(model_sim, test_sim)

    