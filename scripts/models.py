seed_value = 0
import os

os.environ['PYTHONHASHSEED'] = str(seed_value)

from numpy.random import seed

seed(1)

from typing import Tuple

import random

random.seed(seed_value)
import numpy as np

np.random.seed(seed_value)
import tensorflow as tf

tf.compat.v1.set_random_seed(seed_value)
tf.compat.v1.random.set_random_seed(seed_value)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

from keras.models import Sequential
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers.core import Dense, Dropout, Activation

from .metrics import print_metrics, f1, plot_confusion_matrix

from keras.utils.vis_utils import plot_model


def get_model_definition(X_train_shape: Tuple[int, int]) -> Sequential:
    """
    Define the model architecture for training as obtained by hyperparameter search

    :param X_train_shape: Shape of the input data
    :return: Keras model definition
    """
    model = Sequential(name='CNN')

    model.add(Conv1D(8, kernel_size=5, padding='same',
                     input_shape=(X_train_shape[1], X_train_shape[2]), activation='relu'))
    model.add(Conv1D(32, kernel_size=3, activation='relu',
                     input_shape=(X_train_shape[1], X_train_shape[2])))
    model.add(Dropout(0.1294929579149945))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', metrics=[f1],
                  optimizer='sgd')

    return model


def train_model(model, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,
                verbose: bool = True, weighted: bool = False) -> \
        Tuple[Sequential, dict]:
    """
    Train the model

    :param model: Model architecture to train
    :param X_train: Training data
    :param X_test: Testing data
    :param y_train: Training labels
    :param y_test: Testing labels
    :param verbose: Logging Verbosity
    :param weighted: If set to false, class weights will be applied
    :return model: Trained model
    :return report: Classification report containing the main metrics
    """

    if weighted == True:
        class_weights = {0: 3.2, 1: 1}
    else:
        class_weights = {0: 1, 1: 1}
    print(class_weights)
    if verbose:
        print(model.summary())
    model.fit(X_train, y_train, epochs=20, batch_size=32, class_weight=class_weights,
              verbose=True, shuffle=False, validation_split=0.1)
    report = print_metrics(model, X_test, y_test)
    return model, report


def run_sequence_classifications(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,
                                 verbose: bool = False, weighted: bool = False):
    """
    Create, train, evaluate, and save the model

    :param X_train: Training data
    :param X_test: Testing data
    :param y_train: Training labels
    :param y_test: Testing labels
    :param verbose: Logging Verbosity
    :param weighted: If set to false, class weights will be applied
    """
    model_definition = get_model_definition(X_train.shape)

    architecture_plot_path = './figures/architecture.png'
    plot_model(model_definition, to_file=architecture_plot_path, show_shapes=True, show_layer_names=True)
    model, report = train_model(model_definition, X_train, X_test, y_train, y_test, verbose, weighted)
    y_pred = model.predict(X_test)
    fig = plot_confusion_matrix(y_test, y_pred)
    model.save('CNN_Model')
