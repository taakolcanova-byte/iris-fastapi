import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pickle

print("Загрузка данных...")
iris = load_iris()
X = iris.data
y = iris.target

# Масштабирование данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Сохраняем скейлер
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Создание и обучение модели...")
model = Sequential([
    Dense(16, activation='relu', input_shape=(4,)),
    Dense(8, activation='relu'),
    Dense(3, activation='softmax') # 3 класса (сорта ириса)
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

model.fit(X_scaled, y, epochs=100, verbose=0)

# Сохраняем модель
model.save('iris_model.h5')
print("Готово! Модель 'iris_model.h5' и скейлер 'scaler.pkl' сохранены.")