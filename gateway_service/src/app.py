import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api import ml_router


def init_base_application() -> FastAPI:

    @asynccontextmanager
    async def application_lifespan(application: FastAPI):
        # on start up model initialization
        await asyncio.sleep(2)
        yield
        # on shut down
        await asyncio.sleep(2)

    application: FastAPI = FastAPI(lifespan=application_lifespan)
    return application


def create_application() -> FastAPI:
    application: FastAPI = init_base_application()
    application.include_router(ml_router, prefix='/api/model-a', tags=['model-a'])

    @application.get('/healthcheck')
    async def healthcheck_handler() -> str:
        return 'OK'

    return application


app: FastAPI = create_application()


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)
