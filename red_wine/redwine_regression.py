"""
Using the Red Wine Dataset, from https://www.kaggle.com/piyushgoyal443/red-wine-dataset.

Following my notes from the Udemy course on building Multiple Regression models.
- As I do more stats, I realize how much I've forgotten. This is just practice, and should not be taken seriously.
"""

# -- Import Libraries
import numpy as np
import pandas as pd

# -- Read in the data
df = pd.read_csv("wineQualityReds.csv")
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

# -- Split into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# -- Fit a Linear Regression model to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# -- Predict the test set results
y_pred = regressor.predict(X_test)

# Results
results = {"predicted": y_pred, "actual": y_test}
results = pd.DataFrame(data=results)
print(results)

# -- Building the optimal model using Backward Elimination
# this library doesn't consider the intercept term, correct this by adding a column of 1's to the start of X
import statsmodels.formula.api as sm
X = np.append(arr=np.ones((len(X), 1)).astype(int), values=X, axis=1)

# -- find the most statistically significant independent variables
# first iteration
X_opt = X[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

# second iteration
X_opt = X[:, [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

# third iteration
X_opt = X[:, [0, 2, 3, 4, 5, 6, 7, 9, 10, 11]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

# fourth iteration
X_opt = X[:, [0, 2, 3, 5, 6, 7, 9, 10, 11]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

# fifth iteration
X_opt = X[:, [0, 2, 5, 6, 7, 9, 10, 11]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

# -- found the best features, build a new regression model
X_train, X_test, y_train, y_test = train_test_split(X_opt, y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# we can inspect out test set results
results = {"predicted": y_pred, "actual": y_test}
results = pd.DataFrame(data=results)
print(results)
