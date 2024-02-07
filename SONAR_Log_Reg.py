import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

#Load Data
sonar_df = pd.read_csv("Sonar.csv", header=None)

#Convert to np array
sonar_np = sonar_df.to_numpy()
print(sonar_np.shape[1])

#divide into x and y
sonar_x = sonar_np[:, :sonar_np.shape[1] - 2]
sonar_y = sonar_np[:, sonar_np.shape[1] - 1]
print(sonar_y.shape)
print(sonar_x.shape)

#Label encoding to convert R and M vals to 0 and 1
sonar_y = LabelEncoder().fit_transform(sonar_y)
print(sonar_y)

#Model Selection
model = LogisticRegression()

#CV using KFold
#Create Kfold cross validator
kfold = KFold(n_splits=10, shuffle=True)
cv_score = cross_val_score(model, sonar_x, sonar_y, cv=kfold)
#print(cv_score)
print(f"Accuracy of {cv_score.mean()} with KFold.")
#print(len(cv_score))

#CV using Stratified KFold
skf = StratifiedKFold(n_splits=10, shuffle=True)
skf_cv_score = cross_val_score(model, sonar_x, sonar_y, cv=skf)
print(f"Accuracy of {skf_cv_score.mean()} with Stratified KFold.")

#CV using ShuffleSplit
ss = ShuffleSplit(n_splits=10, test_size=0.3, train_size=0.7)
ss_score = cross_val_score(model, sonar_x, sonar_y, cv=ss)
print(f"Accuracy of {ss_score.mean()} with ShuffleSplit.")





