# MLOps
MLOps course on HSE Master program FTIAD

## Структура репозитория

- `data/`: Этот каталог содержит данные, используемые в проекте.
- `models.py`: Файл с классом Model, в котором реализованы все функции, по которым будем дёргать API
- `api.py`: Реализация fastapi, по которой можно отправлять запросы и получать ответы по моделям
- `README.md`: Этот файл - руководство по применению репозитория.

## Как начать

1. Клонируйте репозиторий на свою локальную машину:

```bash
git clone https://github.com/ZolotarevStat/MLOps.git
```

2. Установите необходимые зависимости и библиотеки:

```bash
pip install -r requirements.txt
```

## HW2

1) Качаем в терминале необходимые штуки
- ```brew install minio/stable/minio```
- ```brew install minio/stable/mc```
- ```brew install dvc```

2) Запускаем ```sh init.sh``` или ```docker-compose build``` -> ```docker-compose up```
3) Тестим работу сервиса, используя курлы по документации функций из ```api.py```

## HW3

1. Добавил unit тесты в папке ```tests/``` с фикстурой на проверку обращения к бакетам в minio, а также проверку получения данных по ссылке
2. Добавил CI/CD 