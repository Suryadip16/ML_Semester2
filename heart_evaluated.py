from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np


#Load Data:

heart_df = pd.read_csv("heart.csv")
print(heart_df.columns)
x = heart_df.drop(["target"], axis=1)
y = heart_df["target"]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.3, shuffle=True)
clf = LogisticRegression(max_iter=300)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
con_matrix = confusion_matrix(y_pred=y_pred, y_true=y_test)
print(con_matrix)

