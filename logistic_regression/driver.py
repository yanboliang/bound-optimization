import pandas as pd
import numpy as np
from regression import LogisticRegression

df = pd.read_csv("/Users/yliang/data/trunk0/spark/assembly/target/tmp/LogisticRegressionSuite/binaryDataset/part-00000", header = None)
X = np.array(df[df.columns[2:6]])
y = np.array(df[df.columns[0]])
lir = LogisticRegression(fit_intercept=True, alpha=1.37, max_iter=100, tol=1e-06, standardization=False)
lir.fit(X, y)
print("coefficients = " + str(lir.coef_))
print("intercept = " + str(lir.intercept_))