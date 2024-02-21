from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import pandas as pd

breast_df = pd.read_csv("wisconsin_breast_cancer.csv")
breast_y = breast_df["diagnosis"]
breast_x = breast_df.drop(["diagnosis"], axis=1)
x_train, x_test, y_train, y_test = train_test_split(breast_x, breast_y, test_size=0.3, shuffle=True)
label = LabelEncoder()
label.fit(y_train)
y_train = label.transform(y_train)
y_test = label.transform(y_test)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

max_depth = range(1, 33)
score = 0
for depth in max_depth:
    model = DecisionTreeClassifier(random_state=36, max_depth=depth)
    cv_scores = cross_val_score(model, x_train, y_train, cv=10)
    mean_score = cv_scores.mean()

    if mean_score > score:
        score = mean_score
        depth_val = depth

clf = DecisionTreeClassifier(random_state=36, max_depth=depth_val)
clf.fit(x_train, y_train)
plt.figure(figsize=(15, 10))
plot_tree(clf, class_names=['Benign', 'Malignant'])
plt.show()


y_pred = clf.predict(x_test)
acc_score = accuracy_score(y_test, y_pred)

print(f"Final accuracy: {acc_score}")






