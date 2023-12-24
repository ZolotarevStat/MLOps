from fastapi.testclient import TestClient
from unittest.mock import patch
from src.api import app
import pytest

client = TestClient(app)

def test_add_model_success():
    # Arrange
    request_dict = {
        "path": "https://raw.githubusercontent.com/ZolotarevStat/University/main/%5BFTIAD%5D%20MLOps/heart.csv",
        "random_seed": 42,
        "for_train_only": 1
                   }
    expected_response = {"message": "Добавили данные",
                         "path": "'https://raw.githubusercontent.com/ZolotarevStat/University/main/%5BFTIAD%5D%20MLOps/heart.csv'"}
    
    with patch('src.models.Model.get_data') as get_data:
        response = client.post("/get_data", json=request_dict)
        get_data.assert_called_once_with(request_dict['path'], 
                                         request_dict['random_seed'],
                                         request_dict['for_train_only'])
        assert response.status_code == 200
        assert response.json() == expected_response


