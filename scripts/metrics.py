from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from keras import backend as K, Sequential
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score


def print_metrics(model: Sequential, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Print main classification metrics (f1, roc auc,...)

    :param model: Model used for predictions
    :param X_test: Test split
    :param y_test: Test split
    :return: Classification Report Dictionary
    """
    y_pred = model.predict(X_test)
    y_pred_l = (y_pred > 0.5)

    print(classification_report(y_test, y_pred_l))
    print('{0: >12}{1: >11}'.format('roc auc', "{0:.2}".format(roc_auc_score(y_test, y_pred_l))))
    return classification_report(y_test, y_pred_l, output_dict=True)


def f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculation of F1-score.

    :param y_true: ground-truth labels
    :param y_pred: predicted labels
    :return: F1-Score
    """

    def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Batch-wise average of recall metric.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Batch-wise average of Precision metric
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def plot_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray,
                          classes: List[str] = ["SNF", "other"]) -> plt.Figure:
    """
    Plot Confusion Matrix with normalized and absolute values.

    :param y_test: ground-truth labels in the test dataset
    :param y_pred: predicted y-values
    :param classes: Textual descriptions of the classes
    :return: Figure
    """
    y_pred_l = y_pred > 0.5

    cm = confusion_matrix(y_test, y_pred_l)
    normalized_values = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    flat_values = ["{0:0.0f}".format(value) for value in cm.flatten()]
    percentages = ["{0:.1%}".format(value) for value in normalized_values.flatten()]

    labels = [f"{v1}\n{v2}" for v1, v2 in zip(flat_values, percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    fig, ax = plt.subplots()
    sns.heatmap(normalized_values, annot=labels, fmt='', cmap='Blues', xticklabels=classes, yticklabels=classes)

    ax.set(ylabel='True label', xlabel='Predicted label', title='Confusion matrix')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.yticks(rotation=0)

    fig.set_size_inches(5, 4.3)
    fig.tight_layout()
    return fig


def create_shap_plot(model: Sequential, X_train: np.ndarray, X_test: np.ndarray, feature_names: np.ndarray):
    """
    Creation of Shap-plot for the input model.

    :param model: Model used for explainability analysis
    :param X_train: Train split
    :param X_test: Test split
    :param feature_names: Feature names
    :return: Shap Plot
    """
    explainer = shap.DeepExplainer(model, X_train)
    shap_values = explainer.shap_values(X_test)

    feature_values = pd.DataFrame(data=np.mean(X_test, axis=1), columns=feature_names)
    shap_plot = shap.summary_plot(np.mean(shap_values[0], axis=1), feature_values)
    return shap_plot
