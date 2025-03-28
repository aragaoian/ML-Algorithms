import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from Utils.evaluation_metrics import EvaluationMetrics
from Supervised.regression import (
    LinearRegression,
    LassoRegression,
    RidgeRegression,
    ElasticNetRegression,
)


def evaluate_model(name, X_test, y_test, y_pred):
    metrics = EvaluationMetrics(n=X_test.shape[0], y=y_test, y_pred=y_pred)
    print(
        f"{name} - MAE: {metrics.mae()}, MSE: {metrics.mse()}, "
        f"RMSE: {metrics.rmse()}, R2: {metrics.rsquared()}"
    )


def main():
    X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    lr = LinearRegression(1000, 0.01)
    lr.gradientDescent(X_train, y_train)
    y_pred_gd = lr.predict(X_test)
    evaluate_model("Gradient Descent Linear Regression", X_test, y_test, y_pred_gd)

    lr.ordinaryleastSquares(X_train, y_train)
    y_pred_ols = lr.predict(X_test)
    evaluate_model(
        "Ordinary Least Squares Linear Regression", X_test, y_test, y_pred_ols
    )

    lr.moorePenroseLeastSquares(X_train, y_train)
    y_pred_mpls = lr.predict(X_test)
    evaluate_model("Moore-Penrose Least Squares", X_test, y_test, y_pred_mpls)

    lr.moorePenroseLeastSquaresSVD(X_train, y_train)
    y_pred_svd = lr.predict(X_test)
    evaluate_model("Moore-Penrose Least Squares (SVD)", X_test, y_test, y_pred_svd)

    lasso = LassoRegression(1000, 0.01, alpha=0.01)
    lasso.fit(X_train, y_train)
    y_pred_lasso = lasso.predict(X_test)
    evaluate_model("Lasso Regression", X_test, y_test, y_pred_lasso)

    ridge = RidgeRegression(1000, 0.01, alpha=0.01)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    evaluate_model("Ridge Regression", X_test, y_test, y_pred_ridge)

    elastic_net = ElasticNetRegression(1000, 0.01, alpha=0.01, l1_ratio=0.5)
    elastic_net.fit(X_train, y_train)
    y_pred_elastic_net = elastic_net.predict(X_test)
    evaluate_model("ElasticNet Regression", X_test, y_test, y_pred_elastic_net)


if __name__ == "__main__":
    main()
