import numpy as np
from Utils.weights import Weights


class Regression:
    def __init__(self, n_iter: int, l_rate: float, alpha: float = 0, ratio: float = 0):
        self.n_iter = n_iter
        self.l_rate = l_rate
        self.w = None
        self.b = 0
        self.error = 0
        self.alpha = alpha
        self.ratio = ratio

    def regularize(self):
        return {
            None: 0,
            "L1": self.alpha * np.sign(self.w),
            "L2": 2 * self.alpha * self.w,
            "L1+L2": (self.alpha * np.sign(self.w))
            + (1 - self.ratio) * (2 * self.alpha * self.w),
        }

    def predict(self, X):
        return np.dot(X, self.w) + self.b

    def fit(self, X, y, reg_factor: str = None):
        n_samples, n_features = X.shape
        if self.w is None:  # persisting weights
            self.w = Weights(n_features).random_uniform()
        y = np.reshape(y, (-1, 1))  # transform an array to a column vector
        regularization = self.regularize()

        for _ in range(self.n_iter):
            y_pred = self.predict(X)
            self.error = y - y_pred
            grad_w = -(np.dot(X.T, self.error) / n_samples) + regularization.get(
                reg_factor, 0
            )
            grad_b = -(np.sum(self.error)) / n_samples
            self.w -= self.l_rate * grad_w  # l_rate * (grad_w + regularization_factor)
            self.b -= self.l_rate * grad_b  # l_rate * grad_b


class LinearRegression(Regression):
    def __init__(self, n_iter: int, l_rate: float):
        super().__init__(n_iter, l_rate)

    def printWeights(self):
        print(self.w, self.w.shape)

    def gradientDescent(self, X, y):
        super().fit(X, y)

    def ordinaryleastSquares(self, X, y):
        """
        This algorithm works by making the total of the squares of the errors as
        small as possible. However, this model is sensitive to outliers, adding bias
        and pulling the line towards it.
        """

        y = y.reshape(-1, 1)

        x_mean = np.mean(X)
        y_mean = np.mean(y)

        self.w = sum((X - x_mean) * (y - y_mean)) / sum(np.pow(X - x_mean, 2))
        self.b = y_mean - self.w * x_mean

    def moorePenroseLeastSquares(self, X, y):
        self.b = np.ones((X.shape[0], 1))
        X = np.column_stack((X, self.b))
        y = y.reshape(-1, 1)

        mp = np.linalg.pinv(X.T @ X) @ X.T @ y
        self.w = mp[:-1]
        self.b = mp[-1:]

    def moorePenroseLeastSquaresSVD(self, X, y):
        """
        X.T @ X reduces dimensionality and optimizes computations due to
        X.T @ X being a symetric and positive semi-definite matrix
        (SVD's computational complexity is reduced with non negative singular values)
        """
        self.b = np.ones((X.shape[0], 1))
        X = np.column_stack((X, self.b))
        y = y.reshape(-1, 1)

        U, S, V = np.linalg.svd(X.T @ X)
        S_diag = np.diag(S)
        mp_svd = (U @ np.linalg.pinv(S_diag) @ V.T) @ (X.T @ y)

        self.w = mp_svd[:-1]
        self.b = mp_svd[-1:]


class LassoRegression(Regression):
    def __init__(self, n_iter, l_rate, alpha: float = 0):
        super().__init__(n_iter, l_rate, alpha)

    def fit(self, X, y):
        super().fit(X, y, reg_factor="L1")


class RidgeRegression(Regression):
    def __init__(self, n_iter, l_rate, alpha: float = 0):
        super().__init__(n_iter, l_rate, alpha)

    def fit(self, X, y):
        super().fit(X, y, reg_factor="L2")


class ElasticNetRegression(Regression):
    def __init__(self, n_iter, l_rate, alpha: float = 0, ratio: float = 0):
        super().__init__(n_iter, l_rate, alpha, ratio)

    def fit(self, X, y):
        super().fit(X, y, reg_factor="L1+L2")
