import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# Генерація випадкових даних
m = 100
X = np.linspace(-3, 3, m)
y = 2 * np.sin(X) + np.random.uniform(-0.6, 0.6, m)

# Побудова графіка
plt.scatter(X, y, label='Data')
plt.title('Random Data')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# Перетворення X в стовпець (необхідно для використання у моделі)
X = X.reshape(-1, 1)

# Побудова моделі лінійної регресії
model = LinearRegression()
model.fit(X, y)

# Передбачення значень за допомогою моделі
y_pred = model.predict(X)

# Побудова графіка
plt.scatter(X, y, label='Data')
plt.plot(X, y_pred, color='red', label='Linear Regression')  
plt.title('Linear Regression on Random Data')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# Перетворення X в стовпець (необхідно для використання у моделі)
X = X.reshape(-1, 1)

# Створення поліноміальних ознак
degree = 5  # Задаємо ступінь полінома
poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(X)

# Побудова моделі поліноміальної регресії
poly_model = LinearRegression()
poly_model.fit(X_poly, y)

# Передбачення значень за допомогою моделі
y_poly_pred = poly_model.predict(X_poly)

# Побудова графіка
plt.scatter(X, y, label='Data')
plt.plot(X, y_poly_pred, color='red', label='Polynomial Regression') 
plt.title('Polynomial Regression on Random Data')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# Оцінка якості моделі за допомогою середньоквадратичної помилки (MSE)
mse = mean_squared_error(y, y_poly_pred)
print(f'Serendiose squared error: {mse}')