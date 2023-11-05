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
    def __init__(self, path, model_type, model_args={}, random_seed=42, for_train_only=True):
        np.random.seed(random_seed)
        self.for_train_only = for_train_only
        self.modelType = None
        self.isTrained = None

        try:
            self.data = pd.read_csv(path, sep=",")
        except Exception as e:
            raise ModelInternalError(message="Проблемы с чтением файла! Используйте ',' в качестве разделителя в файле")

        # self.data.drop(columns=["total sulfur dioxide"], inplace=True)
        if for_train_only:
            self.__xtrain = self.__scaler.fit_transform(self.data.drop(self.data.columns[-1]))
            self.__ytrain = self.data[self.data.columns[-1]]
        else:
            self.__xtrain, self.__xtest, self.__ytrain, self.__ytest = train_test_split(
                self.data.drop(self.data.columns[-1]),
                self.data[self.data.columns[-1]],
                test_size=0.3)

            self.__scaler = MinMaxScaler()
            self.__xtrain = self.__scaler.fit_transform(self.__xtrain)
            self.__xtest = self.__scaler.transform(self.__xtest)

        if model_type == "catboost":
            try:
                self.model = CatBoostClassifier(**model_args)
                self.modelType = "catboost"
                self.isTrained = False
            except Exception as e:
                raise ModelInternalError(
                    message=
                    f"Ошибка при инициализации модели! Суть:\n{getattr(e, 'message', repr(e))}"
                )
        elif model_type == "svc":
            try:
                self.model = SVC(**model_args)
                self.modelType = "svc"
                self.isTrained = False
            except Exception as e:
                raise ModelInternalError(
                    message=
                    f"Ошибка при инициализации модели! Суть:\n{getattr(e, 'message', repr(e))}"
                )
        elif model_type == "logreg":
            try:
                self.model = LogisticRegression(**model_args)
                self.modelType = "logreg"
                self.isTrained = False
            except Exception as e:
                raise ModelInternalError(
                    message=
                    f"Ошибка при инициализации модели! Суть:\n{getattr(e, 'message', repr(e))}"
                )
        else:
            raise ModelInternalError(
                message=
                f"Допустимые типы модели: ['catboost', 'svc', 'logreg']! А получили вот такое - {model_type}.")

    def train(self):
        __start = time.time()
        self.model.fit(self.__xtrain, self.__ytrain)
        self.isTrained = True
        return f"Ваша модель была успешно обучена за {round(time.time() - __start, 2)} секунд!"

    def predict(self, data: dict):
        def prepare_data(dict_data):
            prepared_data = self.__scaler.transform(pd.DataFrame(dict_data))
            assert len(prepared_data.columns) == len(self.__xtrain.columns)
            return prepared_data

        if self.isTrained:
            try:
                return self.model.predict(prepare_data(data))
            except Exception as e:
                raise ModelInternalError(
                    message=
                    f"Ошибка при попытке сделать прогнозы! Суть:\n{getattr(e, 'message', repr(e))}"
                )
        else:
            raise ModelInternalError(message="Модель не обучена!")

    def score(self):
        if self.for_train_only:
            return {
                'train_accuracy': self.model.score(self.__xtrain, self.__ytrain),
                'train_f1': f1_score(self.__xtrain, self.__ytrain)
            }
        else:
            return {
                'train_accuracy': self.model.score(self.__xtrain, self.__ytrain),
                'train_f1': f1_score(self.__xtrain, self.__ytrain),
                'test_accuracy': self.model.score(self.__xtest, self.__ytest),
                'test_f1': f1_score(self.__xtest, self.__ytest)
            }
