from random import randint

from sklearn.linear_model import LinearRegression

TRAIN_SET_LIMIT = 10000
TRAIN_SET_COUNT = 100

TRAIN_INPUT = list()
TRAIN_OUTPUT = list()
for i in range(TRAIN_SET_COUNT):
    a = randint(0, TRAIN_SET_LIMIT)
    b = randint(0, TRAIN_SET_LIMIT)
    c = randint(0, TRAIN_SET_LIMIT)
    op = a + (2*b) + (3*c)
    TRAIN_INPUT.append([a, b, c])
    TRAIN_OUTPUT.append(op)


predictor = LinearRegression(n_jobs=-1)
predictor.fit(X=TRAIN_INPUT, y=TRAIN_OUTPUT)


X_TEST = [[150, 20, 30]]
outcome = predictor.predict(X=X_TEST)
coefficients = predictor.coef_

print('Outcome : {}\nCoefficients : {}'.format(outcome, coefficients))


# FROM: https://towardsdatascience.com/simple-machine-learning-model-in-python-in-5-lines-of-code-fe03d72e78c6

# This python program can predict the outcome of the equation [(a + 2b + 3c) = y]
# It predicts the y-value, rather than calculating it.


# The predictor is a linear regression, needing x and y values.
# [x, values] The predictor is given an input list of random numbers for [a, b, and c].
# [y, values] The predictor is also given the y value (listed as op for output above) for each of these random [a, b, c] arrays

# With this, the predictor now has a map of various outcomes,
# but not all outcomes because there are a set number of iterations (Train_count),
# and a set number of random values that can be assigned to a, b, and c.


# Lastly, the predictor is only given the X value ([a,b,c] array) for its,
# linear regression and the outcome will be what the predictor predicts
# with this given value based on the map that has been fitted for it.

# (does it predict coefficients??)
