# Authors: Yanbo Liang ybliang8@gmail.com

import numpy as np

from scipy import optimize, sparse


def _linear_loss_and_gradient(w, X, y, alpha, sample_weight=None):
	# Note: Only handle w/o intercept currently.
	
	_, n_features = X.shape
	fit_intercept = (n_features + 1 == w.shape[0])
	if fit_intercept:
		intercept = w[-1]
	print("coefficients = " + str(w))
	w = w[:n_features]
	n_samples = np.sum(sample_weight)
	
	diff = np.dot(X, w) - y
	if fit_intercept:
		linear_loss += intercept
	
	loss = 0.5 * np.dot(diff, diff) / n_samples + 0.5 * alpha * np.dot(w, w)
	
	if fit_intercept:
		grad = np.zeros(n_features + 1)
	else:
		grad = np.zeros(n_features)
	
	grad[:n_features] += np.dot(X.T, diff) / n_samples + alpha * w 
	
	print("loss = " + str(loss))
	print("grad = " + str(grad))
	return loss, grad


class LinearRegressor():
	
	def __init__(self, max_iter=100, alpha=0.0001, lower_bound=-np.inf, upper_bound=np.inf, fit_intercept=False, tol=1e-06):
		self.max_iter = max_iter
		self.alpha = alpha
		self.lower_bound=lower_bound
		self.upper_bound=upper_bound
		self.fit_intercept = fit_intercept
		self.tol = tol
	
	def fit(self, X, y, sample_weight=None):
		if sample_weight is not None:
			sample_weight = np.array(sample_weight)
			check_consistent_length(y, sample_weight)
		else:
			sample_weight = np.ones_like(y)
		
		stdy = np.std(y)
		stdX = np.std(X, axis=0)

		y = y / stdy
		X = X / stdX
		
		if self.fit_intercept:
			parameters = np.zeros(X.shape[1] + 1)
		else:
			parameters = np.zeros(X.shape[1])
		#bounds = np.tile([-np.inf, np.inf], (parameters.shape[0], 1))
		bounds = np.zeros([parameters.shape[0], 2])
		
		for i in range(0, parameters.shape[0]):
			bounds[i, 0] = self.lower_bound
			bounds[i, 1] = self.upper_bound
			bounds[i, 0] = bounds[i, 0] * stdX[i] / stdy
			bounds[i, 1] = bounds[i, 1] * stdX[i] / stdy
		try:
			parameters, f, dict_ = optimize.fmin_l_bfgs_b(
				_linear_loss_and_gradient, parameters,
				args=(X, y, self.alpha / stdy, sample_weight),
				maxiter=self.max_iter, tol=self.tol, bounds=bounds,
				iprint=0)
		except TypeError:
			parameters, f, dict_ = optimize.fmin_l_bfgs_b(
				_linear_loss_and_gradient, parameters,
				args=(X, y, self.alpha / stdy, sample_weight),
				bounds=bounds)
		
		self.n_iter_ = dict_.get('nit', None)
		if self.fit_intercept:
			self.intercept_ = parameters[-1]
		else:
			self.intercept_ = 0.0
		self.coef_ = parameters[:X.shape[1]] * stdy / stdX
		
		return self


lor = LinearRegressor(100, 0.05, -np.inf, np.inf, False) 
X = np.array([[1,2],[-1.5,2.1],[-2.1,-2]])
y = np.array([3, 1, 0])
model = lor.fit(X, y)
print(model.coef_)


