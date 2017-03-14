# Authors: Yanbo Liang ybliang8@gmail.com

import numpy as np

from scipy import optimize, sparse

from sklearn.utils import check_X_y
from sklearn.utils import check_consistent_length


def _linear_loss_and_gradient(w, X, y, alpha, sample_weight=None):
    
    # Note: Only handle w/o intercept currently.

    _, n_features = X.shape
    fit_intercept = (n_features + 1 == w.shape[0])
    if fit_intercept:
        intercept = w[-1]
    print("coefficients = " + str(w))
    w = w[:n_features]
    n_samples = np.sum(sample_weight)

    linear_loss = np.dot(X, w) - y
    if fit_intercept:
        linear_loss -= intercept

    loss = 0.5 * np.dot(linear_loss, linear_loss) / n_samples + 0.5 * alpha * np.dot(w, w)

    if fit_intercept:
        grad = np.zeros(n_features + 1)
    else:
        grad = np.zeros(n_features)

    grad[:n_features] += np.dot(X.T, linear_loss) / n_samples + alpha * w 
    
    print("loss = " + str(loss))
    print("grad = " + str(grad))
    return loss, grad


class LinearRegressor():

    def __init__(self, max_iter=100, alpha=0.0001, fit_intercept=True, tol=1e-05):
        self.max_iter = max_iter
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.tol = tol

    def fit(self, X, y, sample_weight=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X.

        sample_weight : array-like, shape (n_samples,)
            Weight given to each sample.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(
            X, y, copy=False, accept_sparse=['csr'], y_numeric=True)
        if sample_weight is not None:
            sample_weight = np.array(sample_weight)
            check_consistent_length(y, sample_weight)
        else:
            sample_weight = np.ones_like(y)

        if self.fit_intercept:
            parameters = np.zeros(X.shape[1] + 1)
        else:
            parameters = np.zeros(X.shape[1])
        
        # Setting it to be zero might cause undefined bounds hence we set it
        # to a value close to zero.
        bounds = np.tile([-np.inf, np.inf], (parameters.shape[0], 1))
        # bounds[-1][0] = 1e-12

        # Type Error caused in old versions of SciPy because of no
        # maxiter argument ( <= 0.9).
        try:
            parameters, f, dict_ = optimize.fmin_l_bfgs_b(
                _linear_loss_and_gradient, parameters,
                args=(X, y, self.alpha, sample_weight),
                maxiter=self.max_iter, tol=self.tol, bounds=bounds,
                iprint=0)
        except TypeError:
            parameters, f, dict_ = optimize.fmin_l_bfgs_b(
                _linear_loss_and_gradient, parameters,
                args=(X, y, self.alpha, sample_weight),
                bounds=bounds)

        self.n_iter_ = dict_.get('nit', None)
        if self.fit_intercept:
            self.intercept_ = parameters[-1]
        else:
            self.intercept_ = 0.0
        self.coef_ = parameters[:X.shape[1]]

        return self
