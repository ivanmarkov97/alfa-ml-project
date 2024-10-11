# ML microservice communication simple
___

## Description:
Simple FastAPI microservice project

### model_service
Сервис для обучения и serving ML-модели на основе RandomForest

### gateway_service
Сервис для приема http-запросов и отправки запросов к ML-модели

## Запуск через CLI
```commandline
docker-compose up --build
```

Structure:
```
├── README.md
├── docker-compose.yaml
├── gateway_service
│        ├── Dockerfile
│        ├── pyproject.toml
│        └── src
│            ├── __init__.py
│            ├── api.py
│            ├── app.py
│            └── schema.py
└── model_service
    ├── Dockerfile
    ├── models
    │        └── model.pkl
    ├── pyproject.toml
    └── src
        ├── __init__.py
        ├── app.py
        ├── dataset.py
        ├── metrics.py
        ├── schema.py
        ├── train.py
        └── utils.py
```