import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from Utils.evaluation_metrics import EvaluationMetrics
from Supervised.regression import LinearRegression


def regressions(df_path: str = None, validation_df_path: str = None, target: str = "y"):
    X, y = make_regression(n_samples=500, n_features=1, noise=15, random_state=4)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    lr = LinearRegression(1000, 0.01)
    lr.moorePenroseLeastSquares(X_train, y_train)
    y_pred = lr.predict(X_test)

    # em = EvaluationMetrics(n=X_test.shape[0], y=y_test, y_pred=y_pred)
    # MAE = em.mae()
    # MSE = em.mse()
    # RMSE = em.rmse()
    # RSQUARED = em.rsquared()

    # print(f"MAE: {MAE}")
    # print(f"MSE: {MSE}")
    # print(f"RMSE: {RMSE}")
    # print(f"R2: {RSQUARED}")

    # plt.scatter(X_test, y_test, label="Test Data")
    # plt.plot(X_test, y_pred, color="red", label="Predicted Line")
    # plt.show()


def main():
    regressions()


if __name__ == "__main__":
    main()
