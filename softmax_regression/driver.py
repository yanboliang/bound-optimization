import pandas as pd
import numpy as np
from regression import SoftmaxRegression

df = pd.read_csv("/Users/yliang/data/trunk4/spark/assembly/target/tmp/LogisticRegressionSuite/multinomialDataset/part-00000", header = None)
X = np.array(df[df.columns[2:6]])
y = np.array(df[df.columns[0]])
sample_weight = np.array(df[df.columns[1]])
smr = SoftmaxRegression(fit_intercept=True, alpha=0.1, max_iter=100, tol=1e-06, standardization=False)
smr.fit(X, y, sample_weight)
print("coefficients = " + str(smr.coef_))
print("intercept = " + str(smr.intercept_))