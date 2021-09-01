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

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

script_path = os.path.dirname(__file__)


def preprocess_data(file_name: str = 'AI_HEART_FAILURE_CNN.csv', data_path: str = '../data/',
                    max_sequence_length: int = 6, class_to_predict: str = 'SKILLED NURSING FACILITY', **kwargs) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Index, np.ndarray]:
    """
    Data Preprocessing

    :param file_name: Name of file containing the data
    :param data_path: Path to data
    :param max_sequence_length: Length at which each sequence is cut off / each sequence is padded to
    :param class_to_predict: Class to be predicted. All other classes are aggregated as class OTHER
    :return X_train: Training data
    :return X_test: Testing data
    :return y_train: Training labels
    :return y_test: Testing labels
    :return label_names: Textual description of labels
    :return feature_names: Textual description of features
    """

    relative_file_path = os.path.join(data_path, file_name)
    absolute_file_path = os.path.join(script_path, relative_file_path)

    event_log = prepare_log(absolute_file_path)

    event_log = scale_data(event_log)

    y, label_names = prepare_labels(event_log, class_to_predict)
    y = [item[0] for item in y]
    event_log.drop(['discharge_location'], axis=1, inplace=True)

    X, feature_names = prepare_sequences(event_log, max_sequence_length)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, shuffle=False)
    return X_train, X_test, np.asarray(y_train), np.asanyarray(y_test), label_names, feature_names


def prepare_log(file_path: str) -> pd.DataFrame:
    """
    Prepare log data by removing unnecessary columns, filling NaN values and deriving transfer_duration and Length of Stay (LOS).

    :param file_path: Path to file containing data
    :return event_log: Prepared event log
    """
    event_log = pd.read_csv(file_path, sep=',')
    event_log[['intime', 'outtime', 'dischtime', 'admittime']] = event_log[
        ['intime', 'outtime', 'dischtime', 'admittime']].apply(pd.to_datetime)

    event_log['transfer_duration'] = event_log['outtime'] - event_log['intime']
    event_log['transfer_duration'] = event_log['transfer_duration'].dt.total_seconds()
    event_log['LOS'] = event_log['dischtime'] - event_log['admittime']
    event_log['LOS'] = event_log['LOS'].dt.total_seconds()

    event_log.drop(['intime', 'outtime', 'dischtime', 'admittime', 'subject_id', 'admission_type', 'Unnamed: 0'],
                   axis=1, inplace=True)

    event_log['Creatinine'] = event_log['Creatinine'].fillna(method='bfill')
    event_log['Urea Nitrogen'] = event_log['Urea Nitrogen'].fillna(method='bfill')
    event_log['Hemoglobin'] = event_log['Hemoglobin'].fillna(method='bfill')
    event_log['Glucose'] = event_log['Glucose'].fillna(method='bfill')
    event_log['Red Blood Cells'] = event_log['Red Blood Cells'].fillna(method='bfill')
    event_log['marital_status'] = event_log['marital_status'].fillna('UNKNOWN')

    return event_log


def scale_data(event_log: pd.DataFrame) -> pd.DataFrame:
    """
    Scale all numerical columns to zero mean and unit variance

    :param event_log: Event log containing data
    :return event_log: Event log containing scaled data
    """

    scaler = StandardScaler()
    columns_to_scale = ['med_count', 'LOS', 'transfer_duration', 'lab_count', 'count_icd', 'transfer_age', 'Creatinine',
                        'Urea Nitrogen', 'Hemoglobin', 'Glucose', 'Red Blood Cells']

    for column in columns_to_scale:
        event_log[column] = scaler.fit_transform(event_log[column].values.reshape(-1, 1))

    return event_log


def prepare_labels(event_log: pd.DataFrame, class_to_predict: str) -> Tuple[np.ndarray, pd.Index]:
    """
    Prepare labels for the dataset

    :param event_log: Event log containing data
    :param class_to_predict: Class to be predicted. All other classes are aggregated as class OTHER
    :return dl_output: Labels for the dataset
    :return label_names: Textual description of labels (since they are represented as 0 and 1)
    """
    discharge_locations = event_log[['hadm_id', 'discharge_location']]
    discharge_locations.drop_duplicates(inplace=True)

    if class_to_predict != 'All':
        discharge_locations['discharge_location'] = np.where(
            discharge_locations['discharge_location'] == class_to_predict, class_to_predict, 'OTHER')

    discharge_locations = discharge_locations['discharge_location']
    discharge_locations = pd.get_dummies(discharge_locations, columns=['discharge_location'])
    discharge_locations.reset_index(inplace=True, drop=True)
    dl_output = discharge_locations.to_numpy()
    label_names = discharge_locations.columns

    return dl_output, label_names


def prepare_sequences(event_log: pd.DataFrame, max_sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    One-hot encode all categorical features. Transform the input dataframe to a 3D array containing
    sequences of length 'max_sequence_length' grouped by patient stay ('hadm_id').

    :param event_log: Event log containing data
    :param max_sequence_length: Length at which each sequence is cut off / each sequence is padded to
    :return feature_vector: Dataset with each sequence of length 'max_sequence_length'
    :return feature_names: Textual description of features

    """
    oh_encoded = pd.get_dummies(data=event_log,
                                columns=['concept:name', 'admission_location', 'insurance', 'marital_status',
                                         'ethnicity', 'gender'])

    sequences = oh_encoded.groupby('hadm_id').agg(lambda x: list(x))

    feature_names = sequences.columns.values

    prepared_features = []

    for column in sequences.columns:
        feature = np.array([np.array(xi) for xi in sequences[column]], dtype=object)
        feature = pad_sequences(feature, maxlen=max_sequence_length, dtype='float')
        feature = np.expand_dims(feature, axis=2)
        prepared_features.append(feature)

    feature_vector = np.concatenate(prepared_features, -1)

    return feature_vector, feature_names
