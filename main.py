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


def regressions(df_path: str = None, validation_df_path: str = None, target: str = "y"):
    X, y = make_regression(n_samples=500, n_features=1, noise=15, random_state=4)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    lr = LinearRegression(1000, 0.01)
    lr.moorePenroseLeastSquaresSVD(X_train, y_train)
    y_pred = lr.predict(X_test)

    lasso = LassoRegression(1000, 0.01, alpha=0.01)
    lasso.fit(X_train, y_train)
    y_pred_lasso = lasso.predict(X_test)

    ridge = RidgeRegression(1000, 0.01, alpha=0.01)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)

    # elastic_net = ElasticNetRegression(1000, 0.01, alpha=0.1, l1_ratio=0.5)
    # elastic_net.fit(X_train, y_train)
    # y_pred_elastic_net = elastic_net.predict(X_test)

    em_lasso = EvaluationMetrics(n=X_test.shape[0], y=y_test, y_pred=y_pred_lasso)
    em_ridge = EvaluationMetrics(n=X_test.shape[0], y=y_test, y_pred=y_pred_ridge)
    # em_elastic_net = EvaluationMetrics(n=X_test.shape[0], y=y_test, y_pred=y_pred_elastic_net)

    print(
        f"Lasso Regression - MAE: {em_lasso.mae()}, MSE: {em_lasso.mse()}, RMSE: {em_lasso.rmse()}, R2: {em_lasso.rsquared()}"
    )
    print(
        f"Ridge Regression - MAE: {em_ridge.mae()}, MSE: {em_ridge.mse()}, RMSE: {em_ridge.rmse()}, R2: {em_ridge.rsquared()}"
    )
    # print(f"ElasticNet Regression - MAE: {em_elastic_net.mae()}, MSE: {em_elastic_net.mse()}, RMSE: {em_elastic_net.rmse()}, R2: {em_elastic_net.rsquared()}")

    em = EvaluationMetrics(n=X_test.shape[0], y=y_test, y_pred=y_pred)
    MAE = em.mae()
    MSE = em.mse()
    RMSE = em.rmse()
    RSQUARED = em.rsquared()

    print(f"MAE: {MAE}")
    print(f"MSE: {MSE}")
    print(f"RMSE: {RMSE}")
    print(f"R2: {RSQUARED}")

    # plt.scatter(X_test, y_test, label="Test Data")
    # plt.plot(X_test, y_pred, color="red", label="Predicted Line")
    # plt.show()


def main():
    regressions()


if __name__ == "__main__":
    main()
