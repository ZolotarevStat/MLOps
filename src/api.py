import uvicorn
from fastapi import FastAPI
from pydantic_models import GetDataRequest, AddRequest, TrainRequest, PredictRequest, MetricsRequest, DropRequest

from models import Model

app = FastAPI()
models = Model()

@app.post('/get_data')
def get_data(request: GetDataRequest):
    """
    Запрос, по которому обучаем модель
    Пример: curl -X POST -H "Content-Type: application/json" -d '{"path":"https://raw.githubusercontent.com/ZolotarevStat/University/main/%5BFTIAD%5D%20MLOps/heart.csv",
                                                                  "random_seed": 42,
                                                                  "for_train_only": 0}' http://0.0.0.0:2023/get_data
    Ответ: {"message":"Добавили logreg_model"}
    """
    path = request.path
    random_seed = request.random_seed
    for_train_only = request.for_train_only

    models.get_data(path, random_seed, for_train_only)

    return {"message": "Добавили данные",
            "path": path}


@app.post('/add_model')
def add_model(request: AddRequest):
    """
    Запрос, по которому обучаем модель
    Пример: curl -X POST -H "Content-Type: application/json" -d '{"model_type":"logreg","model_name":"logreg_model", "model_args":{}}' http://0.0.0.0:2023/add_model
    Ответ: {"message":"Добавили logreg_model"}
    """
    model_type = request.model_type
    model_name = request.model_name
    model_args = request.model_args

    models.add_model(model_type, model_name, model_args)

    return {"message": f"Добавили {model_name}"}


@app.get('/models')
@app.post('/models')
def return_models():
    """
    Запрос, по которому возвращаем список объявленных моделей
    Пример: curl -X GET http://0.0.0.0:2023/models
    Ответ: {"message":"{'models':
                            {'catboost': {'cb_model': {'model': <catboost.core.CatBoostClassifier object at 0x13c434fa0>, 'isTrained': False}},
                            'logreg': {'1st_model': {'model': LogisticRegression(), 'isTrained': False}},
                            'svc': {'svc_model': {'model': SVC(), 'isTrained': False}}},
                        'counter': 0}"}
    """
    return {"message": f"{models.get_models()}"}


@app.post("/train")
def train_model(request: TrainRequest):
    """
    Запрос, по которому возвращаем обучаем конкретную модель
    Пример: curl -X POST -H "Content-Type: application/json" -d '{"model_type":"logreg","model_name":"logreg_model"}' http://0.0.0.0:2023/train
    Ответ: {"message":"Обучили logreg_model"}
    """
    model_type = request.model_type
    model_name = request.model_name

    models.train(model_type, model_name)

    return {"message": f"Обучили {model_name}"}


@app.post("/predict")
@app.get("/predict")
def predict_model(request: PredictRequest):
    """
    Запрос, по которому делаем предикт конкретной модели по входному словарю
    Пример: curl -X POST -H "Content-Type: application/json" -d '{"model_type":"logreg","model_name":"1st_model", "data":{"Age":[40], "RestingBP":[140], "Cholesterol":[289], "FastingBS":[0], "MaxHR":[172], "Oldpeak":[0]}}' http://0.0.0.0:2023/predict
    Ответ:{"prediction":"[0]"}
    """
    data = request.data
    model_type = request.model_type
    model_name = request.model_name

    prediction = models.predict(data, model_type, model_name)

    return {"prediction": f"{prediction}"}


@app.post("/metrics")
@app.get("/metrics")
def metrics(request: MetricsRequest):
    """
    Запрос, по которому выводятся метрики для конкретной модели
    Пример: curl -X POST -H "Content-Type: application/json" -d '{"model_type":"logreg","model_name":"logreg_model"}' http://0.0.0.0:2023/metrics
    Ответ: {"message":"Метрики модели  logreg - logreg_model : {'train_accuracy': 0.7741433021806854, 'train_f1': 0.7901591895803183, 'test_accuracy': 0.7717391304347826, 'test_f1': 0.8073394495412844}"}%
    """
    model_type = request.model_type
    model_name = request.model_name

    return {"message": f"Метрики модели  {model_type} - {model_name} : {models.metrics(model_type, model_name)}"}


@app.post("/drop_model")
def drop_model(request: DropRequest):
    """
    Запрос, по которому удаляем конкретную модель
    Пример: curl -X POST -H "Content-Type: application/json" -d '{"model_type":"logreg","model_name":"logreg_model"}' http://0.0.0.0:2023/drop_model
    Ответ: {"message":"Удалили logreg_model"}
    """
    model_type = request.model_type
    model_name = request.model_name

    models.drop_model(model_type, model_name)
    return {"message": f"Удалили {model_name}"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=2023)
