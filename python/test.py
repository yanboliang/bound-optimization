import pandas as pd
import numpy as np
from linear_regression import LinearRegressor

df = pd.read_csv("/Users/yliang/data/trunk1/spark/assembly/target/tmp/LinearRegressionSuite/datasetWithDenseFeature2/part-00000", header = None)
X = df[df.columns[1:3]]
y = np.array(df[df.columns[0]])
regressor = LinearRegressor(fit_intercept=False, alpha=2.3, max_iter=1000, tol=1e-06)
regressor.fit(X, y)
print(regressor.coef_)
print(regressor.intercept_)