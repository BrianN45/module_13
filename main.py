import numpy as np
from random import random
balls = np.arange(1,1000)
maxbin = []
for N in balls:
    bins = np.zeros(N)
    for b in range(N):
        bins[int(N * random())] +=1
    num = 0
    for a in bins:
        if a == 0:
            num += 1
    maxbin.append(num)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(balls.reshape(-1, 1), maxbin)

from sklearn.linear_model import LinearRegression
from sklearn import metrics

mu_regress = LinearRegression()
mu_regress.fit(X=X_train, y=y_train)
predicted = mu_regress.predict(X_test)
expected = y_test
R2 = metrics.r2_score(expected, predicted)
print(mu_regress.coef_, mu_regress.intercept_)

import matplotlib.pyplot as plt
plt.plot(balls, maxbin)
plt.show()