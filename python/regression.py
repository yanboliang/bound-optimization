# Authors: Yanbo Liang ybliang8@gmail.com

import numpy as np
from scipy import optimize, sparse

def _linear_loss_and_gradient(w, X, y, alpha, sample_weight, xStd, standardization):
	
	_, n_features = X.shape
	fit_intercept = (n_features + 1 == w.shape[0])
	if fit_intercept:
		intercept = w[-1]
	print("coefficients = " + str(w))
	w = w[:n_features]
	n_samples = np.sum(sample_weight)
	
	diff = np.dot(X, w) - y
	if fit_intercept:
		diff += intercept

	if standardization:
		l2reg = 0.5 * alpha * np.dot(w, w)
		l2reg_grad = alpha * w
	else:
		_w = w / xStd
		l2reg = 0.5 * alpha * np.dot(_w, _w)
		l2reg_grad = alpha * _w / xStd
	
	loss = 0.5 * np.dot(diff, diff) / n_samples + l2reg
	
	if fit_intercept:
		grad = np.zeros(n_features + 1)
		grad[-1] = np.sum(diff) / n_samples
	else:
		grad = np.zeros(n_features)
	
	grad[:n_features] += np.dot(X.T, diff) / n_samples + l2reg_grad
	
	print("loss = " + str(loss))
	print("grad = " + str(grad))
	return loss, grad

class LinearRegression():
	
	def __init__(self, max_iter=100, alpha=0.0001, lower_bound=None, upper_bound=None,
				 fit_intercept=False, tol=1e-06, standardization=True):
		self.max_iter = max_iter
		self.alpha = alpha
		self.lower_bound=lower_bound
		self.upper_bound=upper_bound
		self.fit_intercept = fit_intercept
		self.tol = tol
		self.standardization = standardization
	
	def fit(self, X, y, sample_weight=None):
		if sample_weight is not None:
			sample_weight = np.array(sample_weight)
			check_consistent_length(y, sample_weight)
		else:
			sample_weight = np.ones_like(y)
		
		yStd = np.std(y)
		xStd = np.std(X, axis=0)

		y = y / yStd
		X = X / xStd
		self.alpha = self.alpha / yStd

		n_features = X.shape[1]
		
		if self.fit_intercept:
			n_parameters = n_features + 1
		else:
			n_parameters = n_features
		
		parameters = np.zeros(n_parameters)
		
		if self.lower_bound is None:
			self.lower_bound = np.full([n_parameters], -np.inf)
		if self.upper_bound is None:
			self.upper_bound = np.full([n_parameters], np.inf)

		bounds = np.zeros([n_parameters, 2])
		for i in range(0, n_parameters):
			bounds[i, 0] = self.lower_bound[i] 
			bounds[i, 1] = self.upper_bound[i]
			if self.standardization:
				if i < n_features:
					bounds[i, 0] = bounds[i, 0] * xStd[i] / yStd
					bounds[i, 1] = bounds[i, 1] * xStd[i] / yStd
				else:
					bounds[i, 0] = bounds[i, 0] / yStd
					bounds[i, 1] = bounds[i, 1] / yStd
		try:
			parameters, f, dict_ = optimize.fmin_l_bfgs_b(
				_linear_loss_and_gradient, parameters,
				args=(X, y, self.alpha, sample_weight, xStd, self.standardization),
				maxiter=self.max_iter, tol=self.tol, bounds=bounds,
				iprint=0)
		except TypeError:
			parameters, f, dict_ = optimize.fmin_l_bfgs_b(
				_linear_loss_and_gradient, parameters,
				args=(X, y, self.alpha, sample_weight, xStd, self.standardization),
				bounds=bounds)
		
		self.n_iter_ = dict_.get('nit', None)
		if self.fit_intercept:
			self.intercept_ = parameters[-1] * yStd
		else:
			self.intercept_ = 0.0
		self.coef_ = parameters[:n_features] * yStd / xStd
		
		return self

# Used for testing
# X = np.array([[1,2],[-1.5,2.1],[-2.1,-2]])
# y = np.array([3, 1, 0])
# lir = LinearRegression(100, 0.05, -np.inf, np.inf, False) 
# lir.fit(X, y)
# print(lir.coef_)