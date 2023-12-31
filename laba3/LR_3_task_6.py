import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Генерація випадкових даних
m = 100
X = np.linspace(-3, 3, m)
y = 2 * np.sin(X) + np.random.uniform(-0.6, 0.6, m)

# Перетворення X в стовпець (необхідно для використання у моделі)
X = X.reshape(-1, 1)

# Побудова моделі лінійної регресії
model = LinearRegression()
train_sizes, train_scores, validation_scores = learning_curve(model, X, y, train_sizes=[0.1, 0.3, 0.5, 0.7, 0.9], cv=5)

# Обчислення середніх значень помилок для тренування та тестування
train_scores_mean = np.mean(train_scores, axis=1)
validation_scores_mean = np.mean(validation_scores, axis=1)

# Побудова кривих навчання
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, label='Training')
plt.plot(train_sizes, validation_scores_mean, label='Testing')
plt.title('Learning Curves for Linear Regression')
plt.xlabel('Training Set Size')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.show()

# Створення поліноміальних ознак
degree = 2  # Задаємо ступінь полінома
poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(X)

# Побудова моделі поліноміальної регресії
poly_model = LinearRegression()
train_sizes_poly, train_scores_poly, validation_scores_poly = learning_curve(poly_model, X_poly, y, train_sizes=[0.1, 0.3, 0.5, 0.7, 0.9], cv=2)

# Обчислення середніх значень помилок для тренування та тестування
train_scores_mean_poly = np.mean(train_scores_poly, axis=1)
validation_scores_mean_poly = np.mean(validation_scores_poly, axis=1)

# Побудова кривих навчання для поліноміальної регресії
plt.figure(figsize=(10, 6))
plt.plot(train_sizes_poly, train_scores_mean_poly, label='Training (Polynomial)')
plt.plot(train_sizes_poly, validation_scores_mean_poly, label='Testing (Polynomial)')
plt.title('Learning Curves for Polynomial Regression (Degree {})'.format(degree))
plt.xlabel('Training Set Size')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.show()
