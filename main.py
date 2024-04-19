# CS2023 - Lab13

_author_ = "Brian Nguyen"
_credits_ = ["N/A"]
_email_ = "nguyeb2@mail.uc.edu"

import numpy as np
from random import random

# Create data
balls = np.arange(1, 1000)
maxbin = []
for N in balls:
    bins = np.zeros(N)
    for b in range(N):
        bins[int(N * random())] += 1
    # Check which bins have no balls
    num = 0
    for a in bins:
        if a == 0:
            num += 1

    # Add num of 0 bins to data table
    maxbin.append(num)

# Create linear regression model
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(balls.reshape(-1, 1), maxbin)

from sklearn.linear_model import LinearRegression
from sklearn import metrics

mu_regress = LinearRegression()
mu_regress.fit(X=X_train, y=y_train)
predicted = mu_regress.predict(X_test)
expected = y_test
R2 = metrics.r2_score(expected, predicted)
print("SciPy Linear Regression Solution\n slope: {0}\n intercept: {1}\n R2: {2}"
      .format(mu_regress.coef_[0], mu_regress.intercept_, R2))

# Plot data
import matplotlib.pyplot as plt

plt.plot(balls, maxbin)
plt.show()
