# Authors: Yanbo Liang ybliang8@gmail.com

import numpy as np
from scipy import optimize, sparse
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.extmath import (logsumexp, safe_sparse_dot, squared_norm)

def _multinomial_loss_and_gradient(w, X, Y, alpha, sample_weight, xStd, standardization):
	
	#print(str(w))
	_, n_features = X.shape
	_, n_classes = Y.shape
	n_samples = np.sum(sample_weight)
	sample_weight = sample_weight[:, np.newaxis]
	fit_intercept = (w.size == n_classes * (n_features + 1))
	grad = np.zeros((n_classes, n_features + bool(fit_intercept)))

    # Calculate loss value
	w = w.reshape(n_classes, -1)
	
	if fit_intercept:
		intercept = w[:, -1]
		w = w[:, :-1]
	else:
		intercept = 0
	p = safe_sparse_dot(X, w.T)
	p += intercept
	p -= logsumexp(p, axis=1)[:, np.newaxis]

	if standardization:
		_w = w.ravel()
		l2reg = 0.5 * alpha * safe_sparse_dot(_w, _w)
	else:
		_w = w.ravel()
		xStd = np.tile(xStd, n_classes)
		_w = _w / xStd
		l2reg = 0.5 * alpha * squared_norm(_w)

	loss = -(sample_weight * Y * p).sum() + l2reg
	#print("loss = " + str(loss))
	p = np.exp(p, p)

	# Calculate gradient array
	diff = sample_weight * (p - Y)

	if standardization:
		l2reg_grad = alpha * w
	else:
		xStd = np.tile(xStd, n_classes)
		_w = w / xStd
		l2reg_grad = alpha * _w / xStd

	grad[:, :n_features] = safe_sparse_dot(diff.T, X) + l2reg_grad
	#print(str(grad))
	if fit_intercept:
		grad[:, -1] = diff.sum(axis=0)
	return loss, grad.ravel()

class SoftmaxRegression():
	
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

		lbin = LabelBinarizer()
		Y_multi = lbin.fit_transform(y)
		classes = np.unique(y)
		n_classes = classes.size
		
		parameters = np.zeros((n_classes, n_features + bool(self.fit_intercept)), order='C').ravel()
		n_parameters = parameters.size
		
		if self.lower_bound is None:
			self.lower_bound = np.full([n_parameters], -np.inf)
		if self.upper_bound is None:
			self.upper_bound = np.full([n_parameters], np.inf)

		bounds = np.zeros([n_parameters, 2])
		for i in range(0, n_parameters):
			bounds[i, 0] = self.lower_bound[i] 
			bounds[i, 1] = self.upper_bound[i]
			
			colIndex = i % (n_features + bool(self.fit_intercept))
			if colIndex < n_features:
				bounds[i, 0] = bounds[i, 0] * xStd[colIndex]
				bounds[i, 1] = bounds[i, 1] * xStd[colIndex]
			else:
				bounds[i, 0] = bounds[i, 0]
				bounds[i, 1] = bounds[i, 1]
		try:
			parameters, f, dict_ = optimize.fmin_l_bfgs_b(
				_multinomial_loss_and_gradient, parameters,
				args=(X, Y_multi, self.alpha, sample_weight, xStd, self.standardization),
				maxiter=self.max_iter, tol=self.tol, bounds=bounds,
				iprint=0)
		except TypeError:
			parameters, f, dict_ = optimize.fmin_l_bfgs_b(
				_multinomial_loss_and_gradient, parameters,
				args=(X, Y_multi, self.alpha, sample_weight, xStd, self.standardization),
				bounds=bounds)
		
		self.n_iter_ = dict_.get('nit', None)
		parameters = parameters.reshape(n_classes, -1)
		if self.fit_intercept:
			self.intercept_ = parameters[:, -1]
			self.coef_ = parameters[:, :-1]
		else:
			self.intercept_ = 0.0
			self.coef_ = parameters
		
		for i in range(0, n_classes):
			self.coef_[i] = self.coef_[i] / xStd

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