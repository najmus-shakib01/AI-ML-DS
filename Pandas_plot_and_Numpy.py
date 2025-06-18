import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

np.random.seed(42)

num_samples = 200
X = np.linspace(0, 10, num_samples)
y = 2 * X + 3 + np.random.normal(0, 2, num_samples) 

y_nonlinear = 0.5 * X**2 + 1.5 * X + 2 + np.random.normal(0, 3, num_samples)

df = pd.DataFrame({
    'Feature': X,
    'Linear_Target': y,
    'Nonlinear_Target': y_nonlinear
})

print("DataFrame Head:")
print(df.head())
print("\nDescriptive Statistics:")
print(df.describe())

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
df.plot.scatter(x='Feature', y='Linear_Target', title='Linear Relationship', ax=plt.gca())

plt.subplot(1, 2, 2)
df.plot.scatter(x='Feature', y='Nonlinear_Target', title='Nonlinear Relationship', ax=plt.gca(), color='orange')

plt.tight_layout()
plt.show()

X = df[['Feature']]
y_linear = df['Linear_Target']

X_train, X_test, y_train, y_test = train_test_split(X, y_linear, test_size=0.2, random_state=42)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_pred = linear_model.predict(X_test)

print("\nLinear Regression Results:")
print(f"Coefficients: {linear_model.coef_}")
print(f"Intercept: {linear_model.intercept_}")
print(f"Mean squared error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R2 score: {r2_score(y_test, y_pred):.2f}")

plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, color='black', label='Actual')
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Predicted')
plt.title('Linear Regression Results')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend()
plt.show()

y_nonlinear = df['Nonlinear_Target']
X_train, X_test, y_train, y_test = train_test_split(X, y_nonlinear, test_size=0.2, random_state=42)

degree = 2
poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
poly_model.fit(X_train, y_train)

y_pred_poly = poly_model.predict(X_test)

print("\nPolynomial Regression Results:")
print(f"Mean squared error: {mean_squared_error(y_test, y_pred_poly):.2f}")
print(f"R2 score: {r2_score(y_test, y_pred_poly):.2f}")

X_plot = np.linspace(0, 10, 100).reshape(-1, 1)
y_plot = poly_model.predict(X_plot)

plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, color='black', label='Actual')
plt.plot(X_plot, y_plot, color='red', linewidth=3, label='Polynomial Fit')
plt.title('Polynomial Regression Results (Degree=2)')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend()
plt.show()