import pandas as pd
import sklearn.model_selection
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps


def load_data():
    sim_df = pd.read_csv("binary_logistic_regression_data_simulated_for_ML.csv")
    x = sim_df.drop(["disease_status"], axis=1)
    y = sim_df["disease_status"]
    return x, y
    # print(X)
    # print(y)


def split_n_pred(X, y):
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3, train_size=0.7, shuffle=True)
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    acc_score = accuracy_score(y_test, pred)
    #print(acc_score)
    return y_test, pred, y_prob[:, 1]


# Define functions for metrics

def confusion_matrix(y_true, y_pred):
    """
    Compute the confusion matrix.
    Parameters:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
    Returns:
        np.array: Confusion matrix.
    """
    unique_labels = np.unique(np.concatenate((y_true, y_pred)))
    matrix = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)
    for i in range(len(y_true)):
        true_idx = np.where(unique_labels == y_true[i])[0][0]
        pred_idx = np.where(unique_labels == y_pred[i])[0][0]
        matrix[true_idx][pred_idx] += 1
    return matrix


def accuracy(y_true, y_pred):
    """
    Compute the accuracy.
    Parameters:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
    Returns:
        float: Accuracy score.
    """
    cm = confusion_matrix(y_true, y_pred)
    correct = np.sum(cm.diagonal())
    total = np.sum(cm)
    return correct / total


def precision(y_true, y_pred, label):
    """
    Compute the precision for a specific class.
    Parameters:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
        label: Label for which precision is calculated.
    Returns:
        float: Precision score.
    """
    cm = confusion_matrix(y_true, y_pred)
    idx = np.where(np.unique(np.concatenate((y_true, y_pred))) == label)[0][0]
    return cm[idx][idx] / np.sum(cm[:, idx])


def sensitivity(y_true, y_pred, label):
    """
    Compute the sensitivity (true positive rate) for a specific class.
    Parameters:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
        label: Label for which sensitivity is calculated.
    Returns:
        float: Sensitivity score.
    """
    cm = confusion_matrix(y_true, y_pred)
    idx = np.where(np.unique(np.concatenate((y_true, y_pred))) == label)[0][0]
    return cm[idx][idx] / np.sum(cm[idx, :])


def specificity(y_true, y_pred, label):
    """
    Compute the specificity (true negative rate) for a specific class.
    Parameters:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
        label: Label for which specificity is calculated.
    Returns:
        float: Specificity score.
    """
    cm = confusion_matrix(y_true, y_pred)
    idx = np.where(np.unique(np.concatenate((y_true, y_pred))) == label)[0][0]
    true_negatives = np.sum(cm) - np.sum(cm[idx, :]) - np.sum(cm[:, idx]) + cm[idx][idx]
    return true_negatives / (np.sum(cm) - np.sum(cm[:, idx]))


def f1_score(y_true, y_pred, label):
    """
    Compute the F1-score for a specific class.
    Parameters:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
        label: Label for which F1-score is calculated.
    Returns:
        float: F1-score.
    """
    precision_val = precision(y_true, y_pred, label)
    sensitivity_val = sensitivity(y_true, y_pred, label)
    return 2 * (precision_val * sensitivity_val) / (precision_val + sensitivity_val)


def roc_curve(y_true, y_prob, thresholds):
    """
    Compute the ROC curve.
    Parameters:
        y_true (array-like): Ground truth labels.
        y_prob (array-like): Predicted probabilities.
    Returns:
        tuple: (fpr, tpr) - False positive rate and true positive rate.
    """

    tpr = []
    fpr = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tp = cm[1][1]
        fn = cm[1][0]
        fp = cm[0][1]
        tn = cm[0][0]
        tpr.append(tp / (tp + fn))
        fpr.append(fp / (fp + tn))
    return fpr, tpr


def auc(y_true, y_prob, thresholds):
    """
    Compute the Area Under the ROC Curve (AUC).
    Parameters:
        y_true (array-like): Ground truth labels.
        y_prob (array-like): Predicted probabilities.
    Returns:
        float: AUC score.
    """
    fpr, tpr = roc_curve(y_true, y_prob, thresholds)
    return simps(tpr, fpr)

def main():
    X, y = load_data()
    y_true, y_pred, y_prob = split_n_pred(X, y)
    thresholds = np.array([0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0])

    # Calculate metrics
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy(y_true, y_pred)
    prec = precision(y_true, y_pred, 1)
    sens = sensitivity(y_true, y_pred, 1)
    spec = specificity(y_true, y_pred, 1)
    f1 = f1_score(y_true, y_pred, 1)
    fpr, tpr = roc_curve(y_true, y_prob, thresholds)
    auc_score = auc(y_true, y_prob, thresholds)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(auc_score))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

    # Output results
    print("Confusion Matrix:")
    print(cm)
    print("Accuracy:", acc)
    print("Precision (class 1):", prec)
    print("Sensitivity (class 1):", sens)
    print("Specificity (class 1):", spec)
    print("F1-score (class 1):", f1)
    print("AUC:", auc_score)


if __name__ == "__main__":
    main()
