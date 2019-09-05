# From https://towardsdatascience.com/introduction-to-machine-learning-algorithms-linear-regression-14c4e325882a
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Using the sckit library
# using pandas library
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

df_train.dropna(inplace=True)

x_train = df_train['x']
y_train = df_train['y']


x_test = df_test['x']
y_test = df_test['y']


x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)


x_train = x_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)


clf = LinearRegression(normalize=True)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(r2_score(y_test, y_pred))


# Building a model

n = 699
alpha = 0.0001

a_0 = np.zeros((n, 1))
a_1 = np.zeros((n, 1))

epochs = 0
while(epochs < 1000):
    y = a_0 + a_1 * x_train
    error = y - y_train
    mean_sq_er = np.sum(error**2)
    mean_sq_er = mean_sq_er/n
    a_0 = a_0 - alpha * 2 * np.sum(error)/n
    a_1 = a_1 - alpha * 2 * np.sum(error * x_train)/n
    epochs += 1
    if(epochs % 10 == 0):
        print(mean_sq_er)


y_prediction = a_0 + a_1 * x_test
print('R2 Score:', r2_score(y_test, y_prediction))

y_plot = []
for i in range(100):
    y_plot.append(a_0 + a_1 * i)
plt.figure(figsize=(10, 10))
plt.scatter(x_test, y_test, color='red', label='GT')
plt.plot(range(len(y_plot)), y_plot, color='black', label='pred')
plt.legend()
plt.show()
