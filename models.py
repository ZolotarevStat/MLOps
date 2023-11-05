import time
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


class ModelInternalError(Exception):
    def __init__(self, message="Что-то пошло не так!"):
        self.message = message
        super().__init__(self.message)


class Model:
    def __init__(self, path='data/heart.csv', random_seed=42, for_train_only=False):
        np.random.seed(random_seed)
        self.for_train_only = for_train_only
        self.modelType = None
        self.isTrained = None
        self.modelsDict = {'models': {'catboost': {}, 'logreg': {}, 'svc': {}},
                           'counter': 0}

        try:
            self.data = pd.read_csv(path, sep=",")
        except Exception as e:
            raise ModelInternalError(message="Проблемы с чтением файла! Используйте ',' в качестве разделителя в файле")

        if for_train_only:
            self.__scaler = MinMaxScaler()
            self.__xtrain = self.data.drop(columns=self.data.columns[-1])
            self.__xtrain = self.__scaler.fit_transform(self.__xtrain.select_dtypes(include='number'))
            self.__ytrain = self.data[self.data.columns[-1]]
        else:
            self.__xtrain, self.__xtest, self.__ytrain, self.__ytest = train_test_split(
                self.data.drop(columns=self.data.columns[-1]),
                self.data[self.data.columns[-1]],
                test_size=0.3)

            self.__scaler = MinMaxScaler()
            self.__xtrain = self.__scaler.fit_transform(self.__xtrain.select_dtypes(include='number'))
            self.__xtest = self.__scaler.transform(self.__xtest.select_dtypes(include='number'))

    def add_model(self, model_type: str = 'logreg', model_name: str = 'awesome_clf', model_args={}):
        if model_name in self.modelsDict['models'][model_type]:
            raise ModelInternalError(
                message=
                f"Ошибка при инициализации модели! Модель с таким названием уже существует! \n"
                f" Существующие названия: {self.modelsDict['models'][model_type].keys()}"
            )
        if model_type == "catboost":
            try:
                self.modelsDict['models'][model_type] = {model_name:
                                                             {'model': CatBoostClassifier(**model_args),
                                                              'isTrained': False}}
            except Exception as e:
                raise ModelInternalError(
                    message=
                    f"Ошибка при инициализации модели! Суть:\n{getattr(e, 'message', repr(e))}"
                )
        elif model_type == "svc":
            try:
                self.modelsDict['models'][model_type] = {model_name:
                                                             {'model': SVC(**model_args),
                                                              'isTrained': False}}
            except Exception as e:
                raise ModelInternalError(
                    message=
                    f"Ошибка при инициализации модели! Суть:\n{getattr(e, 'message', repr(e))}"
                )
        elif model_type == "logreg":
            try:
                self.modelsDict['models'][model_type] = {model_name:
                                                             {'model': LogisticRegression(**model_args),
                                                              'isTrained': False}}
            except Exception as e:
                raise ModelInternalError(
                    message=
                    f"Ошибка при инициализации модели! Суть:\n{getattr(e, 'message', repr(e))}"
                )
        else:
            raise ModelInternalError(
                message=
                f"Допустимые типы модели: ['catboost', 'svc', 'logreg']! А получили вот такое - {model_type}.")

    def train(self, model_type: str = None, model_name: str = None):
        __start = time.time()
        if model_type is None or model_name is None:
            raise ModelInternalError(
                message="Ошибочка! Укажите model_type и\или model_name!"
            )
        model_to_train = self.modelsDict['models'][model_type][model_name]
        model_to_train['model'].fit(self.__xtrain, self.__ytrain)
        model_to_train['isTrained'] = True
        return f"Ваша модель успешно обучена за {round(time.time() - __start, 2)} секунд!"

    def predict(self, data: dict, model_type: str = None, model_name: str = None):
        def prepare_data(dict_data):
            prepared_data = pd.DataFrame(dict_data)
            prepared_data = self.__scaler.transform(prepared_data.select_dtypes(include='number'))
            assert prepared_data.shape[1] == self.__xtrain.shape[1]
            return prepared_data

        if model_type is None or model_name is None:
            raise ModelInternalError(
                message="Ошибочка! Укажите model_type и\или model_name!"
            )
        model_to_predict = self.modelsDict['models'][model_type][model_name]

        if model_to_predict['isTrained']:
            try:
                return model_to_predict['model'].predict(prepare_data(data))
            except Exception as e:
                raise ModelInternalError(
                    message=
                    f"Ошибка при попытке сделать прогнозы! Суть:\n{getattr(e, 'message', repr(e))}"
                )
        else:
            raise ModelInternalError(message="Модель не обучена!")

    def metrics(self, model_type: str = None, model_name: str = None):
        if model_type is None or model_name is None:
            raise ModelInternalError(
                message="Ошибочка! Укажите model_type и\или model_name!"
            )
        model_to_get_metrics = self.modelsDict['models'][model_type][model_name]
        if self.for_train_only:
            return {
                'train_accuracy': model_to_get_metrics['model'].score(self.__xtrain, self.__ytrain),
                'train_f1': f1_score(model_to_get_metrics['model'].predict(self.__xtrain), self.__ytrain)
            }
        else:
            return {
                'train_accuracy': model_to_get_metrics['model'].score(self.__xtrain, self.__ytrain),
                'train_f1': f1_score(model_to_get_metrics['model'].predict(self.__xtrain), self.__ytrain),
                'test_accuracy': model_to_get_metrics['model'].score(self.__xtest, self.__ytest),
                'test_f1': f1_score(model_to_get_metrics['model'].predict(self.__xtest), self.__ytest)
            }

    def get_models(self):
        return self.modelsDict

    def drop_model(self, model_type: str = None, model_name: str = None):
        if model_type is None or model_name is None:
            raise ModelInternalError(
                message="Ошибочка! Укажите model_type и\или model_name!"
            )
        try:
            del self.modelsDict['models'][model_type][model_name]
            self.modelsDict['counter'] -= 1
        except Exception as e:
            raise ModelInternalError(
                message=
                f"Ошибка при попытке удалить модель! Суть:\n{getattr(e, 'message', repr(e))}"
            )
