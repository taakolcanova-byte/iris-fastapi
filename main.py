from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import numpy as np
from tensorflow.keras.models import load_model
import pickle

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Загружаем модель и скейлер при старте
print("Загрузка модели...")
model = load_model('iris_model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Названия классов
classes = ['Setosa (Щетинистый)', 'Versicolor (Разноцветный)', 'Virginica (Виргинский)']

# 1. Отображение формы (GET запрос)
@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

# 2. Обработка формы (POST запрос)
@app.post("/", response_class=HTMLResponse)
async def predict_iris(
    request: Request,
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...)
):
    # Собираем данные в массив
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Масштабируем
    features_scaled = scaler.transform(features)
    
    # Делаем предсказание
    prediction = model.predict(features_scaled)
    class_id = np.argmax(prediction)
    predicted_class = classes[class_id]
    
    # Возвращаем страницу с результатом
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "result": predicted_class}
    )