from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np

np.random.seed(42)
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svr_linear = SVR(kernel='linear', C=100)
svr_poly = SVR(kernel='poly', C=100, degree=3)
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1)

svr_linear.fit(X_train_scaled, y_train)
svr_poly.fit(X_train_scaled, y_train)
svr_rbf.fit(X_train_scaled, y_train)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

linear_mse, linear_r2 = evaluate_model(svr_linear, X_test_scaled, y_test)
poly_mse, poly_r2 = evaluate_model(svr_poly, X_test_scaled, y_test)
rbf_mse, rbf_r2 = evaluate_model(svr_rbf, X_test_scaled, y_test)

print(f"Linear Kernel - MSE: {linear_mse:.4f}, R2: {linear_r2:.4f}")
print(f"Polynomial Kernel - MSE: {poly_mse:.4f}, R2: {poly_r2:.4f}")
print(f"RBF Kernel - MSE: {rbf_mse:.4f}, R2: {rbf_r2:.4f}")