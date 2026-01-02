import numpy as np

from src.linear_regression_numpy import LinearRegressionNumPy
from src.metrics import mse, rmse, mae, r2_score

# Create fake data that follows a straight line
def generate_data(n=100):
    np.random.seed(42)
    X = np.random.rand(n, 1) * 50
    y = 10 + 3 * X[:, 0] + np.random.randn(n) * 5
    return X, y

X, y = generate_data()

# -----------------------
# Normal Equation
# -----------------------
model_ne = LinearRegressionNumPy()
model_ne.fit_normal_equation(X, y)
preds_ne = model_ne.predict(X)

print("Normal Equation Results")
print("MSE:", mse(y, preds_ne))
print("RMSE:", rmse(y, preds_ne))
print("MAE:", mae(y, preds_ne))
print("R2:", r2_score(y, preds_ne))
print("Weights:", model_ne.theta_.flatten())
print()

# -----------------------
# Gradient Descent
# -----------------------
model_gd = LinearRegressionNumPy(standardize=True)
model_gd.fit_gradient_descent(X, y, lr=0.2, epochs=1500)
preds_gd = model_gd.predict(X)

print("Gradient Descent Results")
print("MSE:", mse(y, preds_gd))
print("RMSE:", rmse(y, preds_gd))
print("MAE:", mae(y, preds_gd))
print("R2:", r2_score(y, preds_gd))
print("Weights:", model_gd.theta_.flatten())
