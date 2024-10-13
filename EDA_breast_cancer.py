import pandas as pd

df = pd.read_csv("breast-cancer.csv")
for col in df.columns:
    print(df[col].unique())


