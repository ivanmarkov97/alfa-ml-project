from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI

from train import RandomForestTrainer
from schema import MLModelInput, MLModelOutput
from utils import get_model_dir


def init_base_application() -> FastAPI:

    @asynccontextmanager
    async def application_lifespan(application: FastAPI):
        # on start up model initialization
        model.load_model(str(get_model_dir() / 'model.pkl'))
        await asyncio.sleep(2)
        yield
        # on shut down
        await asyncio.sleep(2)

    application: FastAPI = FastAPI(lifespan=application_lifespan)
    return application


def create_application() -> FastAPI:
    application: FastAPI = init_base_application()

    @application.get('/healthcheck')
    async def healthcheck_handler() -> str:
        return 'OK'

    @application.post('/score')
    async def model_scoring_handler(model_input: MLModelInput) -> MLModelOutput:
        df_input: pd.DataFrame = pd.DataFrame([input_object.dict() for input_object in model_input.objects])
        predictions: list[float] = model.predict(df_input).tolist()
        await asyncio.sleep(3)  # some hard compute
        return MLModelOutput(**{'predictions': predictions})
    return application


model: RandomForestTrainer = RandomForestTrainer()
app: FastAPI = create_application()

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5001)
