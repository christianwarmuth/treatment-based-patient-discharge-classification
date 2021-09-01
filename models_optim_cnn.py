import warnings

warnings.filterwarnings('ignore')

import os
import random
import numpy as np
import tensorflow as tf

from typing import Tuple

from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential

from paper_icpm.scripts import preprocessing
from paper_icpm.scripts.metrics import print_metrics, f1, plot_confusion_matrix

from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperopt import Trials, STATUS_OK, tpe

# Set all necessary seed values
seed_value = 0
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_random_seed(seed_value)
tf.compat.v1.set_random_seed(seed_value)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)


def model_space_definition(X_train: np.ndarray, y_train: np.ndarray):
    """
    Model & Hyperparameterspace Definition

    :param X_train: Training Data
    :param y_train: Test Data
    :return: Best Model after Hyperparameter Search
    """
    model = Sequential(name='CNN')

    model.add(Conv1D({{choice([8, 16, 32, 64, 128, 264])}}, kernel_size={{choice([2, 3, 4, 5, 6])}}, padding='same',
                     input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu'))
    model.add(Conv1D({{choice([16, 32, 64])}}, kernel_size=3, activation='relu',
                     input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout({{uniform(0, 0.5)}}))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense({{choice([32, 64, 128, 256, 512, 1024, 2048])}}))
    model.add(Activation({{choice(['relu', 'tanh'])}}))
    model.add(Dense({{choice([32, 64, 128, 256, 512, 1024])}}))
    model.add(Activation({{choice(['relu', 'tanh'])}}))
    model.add(Dense({{choice([32, 64, 128, 256, 512])}}))
    model.add(Activation({{choice(['relu', 'tanh'])}}))
    model.add(Dense({{choice([32, 64, 128, 256])}}))
    model.add(Activation({{choice(['relu', 'tanh'])}}))
    model.add(Dense({{choice([32, 64, 128])}}))
    model.add(Activation({{choice(['relu', 'tanh'])}}))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', metrics=[f1],
                  optimizer={{choice(['rmsprop', 'adam', 'sgd'])}})

    # class_weights = class_weight.compute_class_weight('balanced',
    #                                                  np.unique(y_train),
    #                                                  y_train)
    class_weights = {0: 1, 1: 1}
    print(class_weights)
    result = model.fit(X_train, np.array(y_train),
                       batch_size={{choice([32, 64, 128])}},
                       epochs={{choice([10, 20, 40, 80])}},
                       verbose=1,
                       validation_split=0.1, class_weight=class_weights)

    validation_acc = np.amax(result.history['val_f1'])
    print('Best metric of epoch:', validation_acc)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}


def data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Data Preparation

    :return: Tuple of Train & Test data
    """
    X_train, X_test, y_train, y_test, label_names, feature_names = preprocessing.preprocess_data()
    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=model_space_definition,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=40, functions=[f1],
                                          trials=Trials())
    X_train, y_train, X_test, y_test = data()
    print("Evalutation of best performing model:")
    y_pred = best_model.predict(X_test)
    print_metrics(best_model, X_test, y_test)
    fig = plot_confusion_matrix(y_test, (y_pred > 0.5))
    fig.savefig('figures/cnn_confusion.png')
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
