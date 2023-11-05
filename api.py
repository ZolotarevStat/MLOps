from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from models import Model
import uvicorn

app = FastAPI()
models = Model()

class AddRequest(BaseModel):
    model_type: str
    model_name: str
    model_args: dict

class TrainRequest(BaseModel):
    model_type: str
    model_name: str

class PredictRequest(BaseModel):
    model_type: str
    model_name: str
    data: dict

class MetricsRequest(BaseModel):
    model_type: str
    model_name: str

class DropRequest(BaseModel):
    model_type: str
    model_name: str


@app.post('/add_model')
def add_model(request: AddRequest):
    model_type = request.model_type
    model_name = request.model_name
    model_args = request.model_args

    models.add_model(model_type, model_name, model_args)

    return {"message": f"Добавили {model_name}"}

@app.get('/models')
@app.post('/models')
def return_models():
    return {"message": f"{models.get_models()}"}

@app.post("/train")
def train_model(request: TrainRequest):
    model_type = request.model_type
    model_name = request.model_name

    models.train(model_type, model_name)

    return {"message": f"Обучили {model_name}"}


@app.post("/predict")
@app.get("/predict")
def predict_model(request: PredictRequest):
    data = request.data
    model_type = request.model_type
    model_name = request.model_name

    prediction = models.predict(data, model_type, model_name)

    return {"prediction": f"{prediction}"}

@app.post("/metrics")
@app.get("/metrics")
def metrics(request: MetricsRequest):
    model_type = request.model_type
    model_name = request.model_name

    return {"message": f"Метрики модели  {model_type} - {model_name} : {models.metrics(model_type, model_name)}"}

@app.post("/drop_model")
def drop_model(request: DropRequest):
    model_type = request.model_type
    model_name = request.model_name

    models.drop_model(model_type, model_name)
    return {"message": f"Удалили {model_name}"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=2023)