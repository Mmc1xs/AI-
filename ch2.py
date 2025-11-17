import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

s = os.path.join('https://archive.ics.uci.edu', 'ml', 'machine-learning-database', 'iris', 'iris.data')
print('URL:', s)


df = pd.read_csv('iris.data',
                 header=None,
                 encoding='utf-8')

df.tail()

# select setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values
print(X)

# plot data
plt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='x', label='versicolor')

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')

# plt.savefig('images/02_06.png', dpi=300)
plt.show()