import os
import random
import asyncio

from httpx import AsyncClient, Response
from fastapi import APIRouter

from schema import MLPredictionObject, BatchMLRequest, BatchMLResponse

ml_router: APIRouter = APIRouter()


async def generate_random_data(n_objects: int) -> BatchMLRequest:
    objects: list[MLPredictionObject] = []
    for i in range(n_objects):
        _object: MLPredictionObject = MLPredictionObject(**{
            f'param_{i + 1}': random.uniform(0, 10)
            for i in range(10)
        })
        objects.append(_object)
    batch_request: BatchMLRequest = BatchMLRequest(objects=objects)
    return batch_request


@ml_router.post('/random-predict')
async def model_request_handler(n_items: int = 1):
    batch_request: BatchMLRequest = await generate_random_data(n_items)
    ml_service_url_a: str = f"{os.environ['ML_SERVICE_URL_A']}/score"
    ml_service_url_b: str = f"{os.environ['ML_SERVICE_URL_B']}/score"
    async with AsyncClient() as client:
        responses: tuple[Response, Response] = await asyncio.gather(*[
            client.post(ml_service_url_a, json=batch_request.dict(), timeout=10),
            client.post(ml_service_url_b, json=batch_request.dict(), timeout=10),
        ])
    avg_predictions: list[float] = []
    for response in responses:
        avg_predictions.append(response.json()['predictions'])
    avg_predictions = [sum(service_predictions) for service_predictions in zip(*avg_predictions)]
    return BatchMLResponse(predictions=avg_predictions)
