import numpy.linalg
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression

# Set a seed for reproducibility
np.random.seed(42)

# Generate a linear regression dataset with 1 feature
X, y = make_regression(n_samples=100, n_features=10, noise=10, random_state=42)



# sim_data_df = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
# x_df = sim_data_df.drop(["disease_score", "disease_score_fluct"], axis=1)
# y_df = sim_data_df["disease_score"]
# x = x_df.to_numpy()
# y = y_df.to_numpy()
# #print(x)
# #print(y.ndim)


x_trans = X.T
squ = x_trans@X
inverse = np.linalg.inv(squ)
theta = inverse@(x_trans@y)
print(theta)
