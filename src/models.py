import pickle
import time
import warnings

import boto3
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from utils import add_to_dvc

warnings.filterwarnings("ignore")


class ModelInternalError(Exception):
    def __init__(self, message="Что-то пошло не так!"):
        self.message = message
        super().__init__(self.message)


s3_client = boto3.client('s3',
                         endpoint_url='http://127.0.0.1:9000',
                         aws_access_key_id='nVobiQ2pBendB0nGOoxi',
                         aws_secret_access_key='tn7duToNNSKZuarizESJZWcaLwnsxf75K8Wp8J9Z')


class Model:
    def __init__(self):
        """
        Инициализация ключевого класса.
        Здесь сразу происходит предобработка данных, поскольку предполагаем, что все модели будут обучаться на одних и тех же данных.
        В идеале надо бы сделать так, чтобы можно было внутри апи прокидывать файлик и выбирать параметры, но не знаю как сделать это.

        :param path: str - Относительный путь до файлика с данными.
        :param random_seed: int - фиксируем весь возможный рандом, чтобы результаты воспроизводились
        :param for_train_only: bool - бинарный флаг. Если True, то обучение на всём исходном наборе данных, если False - только на 70% рандомных строчек
        """
        self.__ytest = None
        self.__ytrain = None
        self.__xtest = None
        self.__xtrain = None
        self.data = None
        self.for_train_only = None
        self.__scaler = None
        try:
            response = s3_client.get_object(Bucket='zol-hw-mlops', Key="modelsDict.pkl")
            pickle_file = response["Body"].read()
            self.modelsDict = pickle.loads(pickle_file)
        except:
            self.modelType = None
            self.isTrained = None
            self.modelsDict = {'models': {'catboost': {}, 'logreg': {}, 'svc': {}},
                               'counter': 0}

    def get_data(self,
                 path: str = 'https://raw.githubusercontent.com/ZolotarevStat/University/main/%5BFTIAD%5D%20MLOps/heart.csv',
                 random_seed: int = 42,
                 for_train_only: int = 1):
        np.random.seed(random_seed)
        self.for_train_only = for_train_only
        dvc_name = path.split('/')[-1].split('.')[0]
        try:
            self.data = pd.read_csv(path, sep=",")
        except Exception as e:
            raise ModelInternalError(
                message="Проблемы с чтением файла! Используйте ',' в качестве разделителя в файле")

        add_to_dvc(self.data, dvc_name + '_raw')

        if self.for_train_only == 1:
            self.__scaler = MinMaxScaler()
            self.__xtrain = self.data.drop(columns=self.data.columns[-1])
            self.__xtrain = self.__scaler.fit_transform(self.__xtrain.select_dtypes(include='number'))
            self.__ytrain = self.data[self.data.columns[-1]]
            add_to_dvc(self.__xtrain, dvc_name + '_train_data')
            add_to_dvc(self.__ytrain, dvc_name + '_train_target')
        else:
            self.__xtrain, self.__xtest, self.__ytrain, self.__ytest = train_test_split(
                self.data.drop(columns=self.data.columns[-1]),
                self.data[self.data.columns[-1]],
                test_size=0.3)

            self.__scaler = MinMaxScaler()
            self.__xtrain = self.__scaler.fit_transform(self.__xtrain.select_dtypes(include='number'))
            self.__xtest = self.__scaler.transform(self.__xtest.select_dtypes(include='number'))

            add_to_dvc(self.__xtrain, dvc_name + '_train_data')
            add_to_dvc(self.__ytrain, dvc_name + '_train_target')
            add_to_dvc(self.__xtest, dvc_name + '_test_data')
            add_to_dvc(self.__ytest, dvc_name + '_test_target')

    def add_model(self, model_type: str = 'logreg', model_name: str = 'awesome_clf', model_args={}):
        """
        Добавляем модель в наш сервис.
        :param model_type: один из трёх возможных типов [logreg, catboost, svc]
        :param model_name: указываем имя, которое будет храниться в словарике с моделями на сервисе
        :param model_args: гиперпараметры добавляемой модели
        """
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
                self.modelsDict['counter'] += 1
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
                self.modelsDict['counter'] += 1
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
                self.modelsDict['counter'] += 1
            except Exception as e:
                raise ModelInternalError(
                    message=
                    f"Ошибка при инициализации модели! Суть:\n{getattr(e, 'message', repr(e))}"
                )
        else:
            raise ModelInternalError(
                message=
                f"Допустимые типы модели: ['catboost', 'svc', 'logreg']! А получили вот такое - {model_type}.")

        pickle_file = pickle.dumps(self.modelsDict)
        s3_client.put_object(Body=pickle_file, Bucket='zol-hw-mlops', Key="modelsDict.pkl")


def train(self, model_type: str = None, model_name: str = None):
    """
    Обучаем конкретную модель
    :param model_type: специфицируем тип модели
    :param model_name: указываем название модели (должно быть одним из уже объявленных). Под этим же именем заливаются данные в DVC
    :return: возвращает строку, отражающую время обучения модели
    """
    __start = time.time()
    if model_type is None or model_name is None:
        raise ModelInternalError(
            message="Ошибочка! Укажите model_type и\или model_name!"
        )

    if model_name not in self.modelsDict['models'][model_type]:
        raise ModelInternalError(
            message=
            f"Ошибка при прогнозировании модели! Модели с таким названием не существует! \n"
            f" Существующие названия: {self.modelsDict['models'][model_type].keys()}"
        )
    model_to_train = self.modelsDict['models'][model_type][model_name]
    model_to_train['model'].fit(self.__xtrain, self.__ytrain)
    model_to_train['isTrained'] = True

    pickle_file = pickle.dumps(self.modelsDict)
    s3_client.put_object(Body=pickle_file, Bucket='zol-hw-mlops', Key="modelsDict.pkl")

    return f"Ваша модель успешно обучена за {round(time.time() - __start, 2)} секунд!"


def predict(self, data: dict, model_type: str = None, model_name: str = None):
    """
    Получаем предикт по конкретной модели
    :param data: словарь данных для предикта в формате {'feature1':[1, 2, 3], 'feature2':[4, 5, 6]}
    :param model_type: специфицируем тип модели
    :param model_name: указываем название модели (должно быть одним из уже объявленных)
    :return: возвращает прогноз
    """

    def prepare_data(dict_data):
        prepared_data = pd.DataFrame(dict_data)
        prepared_data = self.__scaler.transform(prepared_data.select_dtypes(include='number'))
        assert prepared_data.shape[1] == self.__xtrain.shape[1]
        return prepared_data

    if model_type is None or model_name is None:
        raise ModelInternalError(
            message="Ошибочка! Укажите model_type и\или model_name!"
        )

    if model_name not in self.modelsDict['models'][model_type]:
        raise ModelInternalError(
            message=
            f"Ошибка при прогнозировании модели! Модели с таким названием не существует! \n"
            f" Существующие названия: {self.modelsDict['models'][model_type].keys()}"
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
    """
    Для интереса смотрим метрики конкретной модели
    :param model_type: специфицируем тип модели
    :param model_name: указываем название модели (должно быть одним из уже объявленных)
    :return: возвращает словарик метрик классификации
    """
    if model_type is None or model_name is None:
        raise ModelInternalError(
            message="Ошибочка! Укажите model_type и\или model_name!"
        )
    if model_name not in self.modelsDict['models'][model_type]:
        raise ModelInternalError(
            message=
            f"Ошибка при прогнозировании модели! Модели с таким названием не существует! \n"
            f" Существующие названия: {self.modelsDict['models'][model_type].keys()}"
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
    """
    :return: словарь обученных и доступных для обучения моделей
    """
    return self.modelsDict


def drop_model(self, model_type: str = None, model_name: str = None):
    """
    Удаление конкретной модели
    :param model_type: специфицируем тип модели
    :param model_name: указываем название модели (должно быть одним из уже объявленных)
    """
    if model_type is None or model_name is None:
        raise ModelInternalError(
            message="Ошибочка! Укажите model_type и\или model_name!"
        )
    if model_name not in self.modelsDict['models'][model_type]:
        raise ModelInternalError(
            message=
            f"Ошибка при прогнозировании модели! Модели с таким названием не существует! \n"
            f" Существующие названия: {self.modelsDict['models'][model_type].keys()}"
        )
    try:
        del self.modelsDict['models'][model_type][model_name]
        self.modelsDict['counter'] -= 1
    except Exception as e:
        raise ModelInternalError(
            message=
            f"Ошибка при попытке удалить модель! Суть:\n{getattr(e, 'message', repr(e))}"
        )

    pickle_file = pickle.dumps(self.modelsDict)
    s3_client.put_object(Body=pickle_file, Bucket='zol-hw-mlops', Key="modelsDict.pkl")
