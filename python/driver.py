import pandas as pd
import numpy as np
from regression import LinearRegression

df = pd.read_csv("/Users/yliang/data/trunk1/spark/assembly/target/tmp/LinearRegressionSuite/datasetWithDenseFeature2/part-00000", header = None)
X = np.array(df[df.columns[1:3]])
y = np.array(df[df.columns[0]])
lir = LinearRegression(fit_intercept=False, alpha=2.3, max_iter=1000, tol=1e-06, standardization=False)
lir.fit(X, y)
print("coefficients = " + str(lir.coef_))
print("intercept = " + str(lir.intercept_))