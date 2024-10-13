import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as con_mat
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd


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
    return cm[0][0] / np.sum(cm[0, :])

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
    return cm[0][0] / np.sum(cm[:, 0])

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
    return cm[1][1] / np.sum(cm[1, :])

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

def fpr_tpr(y_true, y_pred):
    """
    Compute the ROC curve.
    Parameters:
        y_true (array-like): Ground truth labels.
        y_prob (array-like): Predicted probabilities.
    Returns:
        tuple: (fpr, tpr) - False positive rate and true positive rate.
    """
    #cm = confusion_matrix(y_true, y_pred)
    tpr = sensitivity(y_true, y_pred)
    fpr = 1 - specificity(y_true, y_pred)
    return fpr, tpr

def auc(y_true, y_prob):
    """
    Compute the Area Under the ROC Curve (AUC).
    Parameters:
        y_true (array-like): Ground truth labels.
        y_prob (array-like): Predicted probabilities.
    Returns:
        float: AUC score.
    """
    fpr, tpr = roc_curve(y_true, y_prob)
    return np.trapz(tpr, fpr)

# Example usage


#Load Data:

heart_df = pd.read_csv("heart.csv")
print(heart_df.columns)
x = heart_df.drop(["target"], axis=1)
y = heart_df["target"]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.3, shuffle=True)
clf = LogisticRegression(max_iter=300)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
probs = clf.predict_proba(x_test)
y_proba = probs[:, 1]

# Generate example data
y_true = np.array([0, 1, 0, 1, 0, 1])
y_pred = np.array([0, 0, 0, 1, 1, 1])
y_prob = np.array([0.2, 0.8, 0.3, 0.7, 0.6, 0.9])

# Calculate metrics
cm = confusion_matrix(y_true, y_pred)
cm1 = con_mat(y_true, y_pred)
print(cm)
print(cm1)
acc = accuracy(y_true, y_pred)
prec = precision(y_true, y_pred, 1)
sens = sensitivity(y_true, y_pred, 1)
spec = specificity(y_true, y_pred, 1)
f1 = f1_score(y_true, y_pred, 1)
fpr, tpr = roc_curve(y_true, y_prob)
auc_score = auc(y_true, y_prob)

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
