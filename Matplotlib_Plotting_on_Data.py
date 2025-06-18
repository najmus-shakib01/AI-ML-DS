import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification, make_regression

X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual Data', alpha=0.7)
plt.plot(X, y_pred, color='red', linewidth=2, label='Linear Regression')
plt.title('Linear Regression Visualization', fontsize=16)
plt.xlabel('Feature', fontsize=14)
plt.ylabel('Target', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()