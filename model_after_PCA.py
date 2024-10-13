import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import xgboost as xgb
from ISLP import load_data

NCI60 = load_data('NCI60')
nci_labels = NCI60['labels']
nci_labels_array = np.array(nci_labels)
print(nci_labels_array)
nci_data = NCI60['data']

scaler = StandardScaler()
scaler.fit(nci_data)
nci_data_scaled = scaler.transform(nci_data)

nci_pca = PCA()
nci_scores = nci_pca.fit_transform(nci_data_scaled)
ticks = np.arange(nci_pca.n_components_) + 1
plt.plot(ticks, nci_pca.explained_variance_ratio_.cumsum(), marker="o")
plt.xlabel("Principal Components")
plt.ylabel("Cumulative Variance Explained")
plt.title("Cumulative Variance Explained by PC")
plt.show()

# encoder = LabelEncoder()
# encoder.fit(y_train)
# y_train = encoder.transform(y_train)
# y_test = encoder.transform(y_test)

class_to_number_mapping = {
    'CNS': 0,
    'RENAL': 1,
    'COLON': 2,
    'LEUKEMIA': 3,
    'MELANOMA': 4,
    'NSCLC': 5,
    'OVARIAN': 6,
    'PROSTATE': 7,
    'K562A-repro': 8,
    'K562B-repro': 8,
    'MCF7A-repro': 8,
    'MCF7D-repro': 8,
    'UNKNOWN': 8

}
encoded_labels = []
for label in nci_labels_array:
    if label in class_to_number_mapping:
        val = class_to_number_mapping[label]
        encoded_labels.append(val)

x_train, x_test, y_train, y_test = train_test_split(nci_scores[:, :51], encoded_labels, test_size=0.3, shuffle=True,
                                                    random_state=42, stratify=encoded_labels)

xg_boost = xgb.XGBClassifier()
lm = LogisticRegression()

xg_boost.fit(x_train, y_train)
y_pred_xgb = xg_boost.predict(x_test)
xgb_acc = accuracy_score(y_test, y_pred_xgb)
print(f"XGB Acc : {xgb_acc}")

lm.fit(x_train, y_train)
y_pred_lm = lm.predict(x_test)
lm_acc = accuracy_score(y_test, y_pred_lm)
print(f"Log Reg Acc : {lm_acc}")
