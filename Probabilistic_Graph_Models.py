import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris_df = pd.DataFrame(load_iris().data)
iris_df['class'] = load_iris().target
print(iris_df)

iris_df.columns = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
iris_df.dropna(how="all", inplace=True)

iris_df = iris_df.drop(iris_df.columns[2:4], axis=1)

print(iris_df)
gaus_noise = np.random.normal(0, 1, 150)
iris_df['sepal_len'] = pd.qcut(iris_df['sepal_len'] + gaus_noise, 4, labels=[0, 1, 2, 3])
iris_df['sepal_wid'] = pd.qcut(iris_df['sepal_wid'] + gaus_noise, 4, labels=[0, 1, 2, 3])
print(iris_df)

dtc = DecisionTreeClassifier(max_depth=2)
dtc.fit(iris_df.drop(['class'], axis=1), iris_df['class'])
y_pred = dtc.predict(iris_df.drop(['class'], axis=1))
print(accuracy_score(iris_df['class'], y_pred))

joint_probs = iris_df.groupby(['sepal_len', 'sepal_wid', 'class'])['class'].count() / 150
print(joint_probs)

only_ft_1 = iris_df.iloc[0:50,:]
only_ft = only_ft_1.drop(['class'], axis=1)
print(len(only_ft))

y_pred = []
for index, row in only_ft.iterrows():
    result = joint_probs[row['sepal_len'], row['sepal_wid']]
    y_pred.append(result.idxmax())

print(accuracy_score(only_ft_1['class'], y_pred))

# for i in range(len(only_ft)):
#     v = only_ft.iloc[i, :]
#     sep_len = v['sepal_len']
#     sep_wid = v['sepal_wid']
#     print(joint_probs.get_group('sepal_len'))
#     #print(joint_probs.loc[(lambda joint_probs:joint_probs['sepal_len'] == sep_len) & (lambda joint_probs: joint_probs['sepal_wid'] == sep_wid)])
