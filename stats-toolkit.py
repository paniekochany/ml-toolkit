import numpy as np
import pandas as pd
from scipy import stats

def simple_linear_regression_parameters_bootstrap_ci(df: pd.DataFrame, number_of_iterations: int,
                                                     confidence_level: float) -> str:
    '''Calculate Confidence Interval for alpha and beta parameters in simple linear regression.

    In df df[:, [0]] is a predictor column, and df[:, 1] a label column.'''

    X = df.iloc[:, [0]]
    y = df.iloc[:, 1]

    lm = LinearRegression()
    lm.fit(X, y)
    y_pred = lm.predict(X)
    errors = y - y_pred

    alphas = []
    betas = []

    for i in range(number_of_iterations):
        bootstrap_errors = np.random.choice(errors, size=200, replace=True)
        X = df.iloc[:, [0]]
        y = lm.predict(X) + bootstrap_errors
        lm_bootstrap = LinearRegression()
        lm_bootstrap.fit(X, y)
        alpha = lm_bootstrap.intercept_
        beta = lm_bootstrap.coef_
        alphas.append(alpha)
        betas.append(beta)

    se_alpha = np.std(alphas)
    se_beta = np.std(betas)

    z = abs(stats.t.ppf(q=(1 - confidence_level) / 2, df=len(X) - 1))

    print(f'alpha: {lm.intercept_} in CI: [{lm.intercept_ - 1.96 * se_alpha}, {lm.intercept_ + 1.96 * se_alpha}]')
    print(f'beta: {lm.coef_[0]} in CI: [{lm.coef_ - z * se_beta}, {lm.coef_ + z * se_beta}]')