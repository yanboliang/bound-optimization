# Authors: Yanbo Liang ybliang8@gmail.com

import numpy as np
from scipy import optimize, sparse

def _logistic_loss_and_gradient(w, X, y, alpha, sample_weight, xStd, standardization):
	
	_, n_features = X.shape
	fit_intercept = (n_features + 1 == w.shape[0])
	if fit_intercept:
		intercept = w[-1]
	#print("coefficients = " + str(w))
	w = w[:n_features]
	n_samples = np.sum(sample_weight)
	
	margin = np.dot(X, w)
	if fit_intercept:
		margin += intercept
	margin = -margin
	
	multiplier = 1.0 / (1.0 + np.exp(margin)) - y

	loss = np.sum(np.log(1.0 + np.exp(margin)) + margin * (y - 1.0)) / n_samples	

	if standardization:
		l2reg = 0.5 * alpha * np.dot(w, w)
		l2reg_grad = alpha * w
	else:
		_w = w / xStd
		l2reg = 0.5 * alpha * np.dot(_w, _w)
		l2reg_grad = alpha * _w / xStd
	
	loss += l2reg
	
	if fit_intercept:
		grad = np.zeros(n_features + 1)
		grad[-1] = np.sum(multiplier) / n_samples
	else:
		grad = np.zeros(n_features)
	
	grad[:n_features] += (np.dot(X.T, multiplier) / n_samples + l2reg_grad)
	
	#print("loss = " + str(loss))
	#print("grad = " + str(grad))
	return loss, grad

class LogisticRegression():
	
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
		
		xStd = np.std(X, axis=0)
		xStd = xStd * np.sqrt(y.size / (y.size - 1.0))

		X = X / xStd

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
			
			if i < n_features:
				bounds[i, 0] = bounds[i, 0] * xStd[i]
				bounds[i, 1] = bounds[i, 1] * xStd[i]
			else:
				bounds[i, 0] = bounds[i, 0]
				bounds[i, 1] = bounds[i, 1]
		try:
			parameters, f, dict_ = optimize.fmin_l_bfgs_b(
				_logistic_loss_and_gradient, parameters,
				args=(X, y, self.alpha, sample_weight, xStd, self.standardization),
				maxiter=self.max_iter, tol=self.tol, bounds=bounds,
				iprint=0)
		except TypeError:
			parameters, f, dict_ = optimize.fmin_l_bfgs_b(
				_logistic_loss_and_gradient, parameters,
				args=(X, y, self.alpha, sample_weight, xStd, self.standardization),
				bounds=bounds)
		
		self.n_iter_ = dict_.get('nit', None)
		if self.fit_intercept:
			self.intercept_ = parameters[-1]
		else:
			self.intercept_ = 0.0
		self.coef_ = parameters[:n_features] / xStd
		
		return self

# Used for testing
# X = np.array([[1.0,2.0],[-1.5,2.1],[-2.1,-2.0]])
# y = np.array([1.0, 1.0, 0.0])
# lor = LogisticRegression(100, 0.5, [-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf], True, 1e-6, False)
# lor.fit(X, y)
# print("coef: %s intercept: %s" % (lor.coef_, lor.intercept_))

# Test _logistic_loss_and_gradient
# sample_weight = np.ones_like(y)
# xStd = np.std(X, axis=0)
# xStd = np.array([1, 1])
# (loss, grad) = _logistic_loss_and_gradient(np.array([1.0, 2.0]), X / xStd, y, 0.0, sample_weight, xStd, True)
# print("loss: %f, grad: %s" % (loss, grad))