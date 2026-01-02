# Implementing-Linear-Regression-with-NumPy

This project shows how **Linear Regression works internally**, built **from scratch using NumPy**.

The goal is to understand:
- how a model learns weights
- how predictions are made
- how error is measured
- how training improves the model

No machine learning libraries are used for the model itself.

---

## What is Linear Regression?

Linear Regression tries to draw the **best straight line** through data.

Example idea:
- If `x` increases, how much does `y` increase?
- Can we predict `y` using one or more inputs?

The model learns this equation:
y = intercept + weight × input

---

## What this project includes

- Linear Regression implemented using **NumPy only**
- Two training methods:
  - **Normal Equation** (direct solution)
  - **Gradient Descent** (step-by-step learning)
- Common evaluation metrics:
  - MSE
  - RMSE
  - MAE
  - R² score
- Simple demo with generated data



