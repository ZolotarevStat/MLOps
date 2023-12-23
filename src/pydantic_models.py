from pydantic import BaseModel


class GetDataRequest(BaseModel):
    path: str
    random_seed: int
    for_train_only: int


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
