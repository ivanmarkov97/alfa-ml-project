from pydantic import BaseModel


class MLPredictionObject(BaseModel):
    param_1: float
    param_2: float
    param_3: float
    param_4: float
    param_5: float
    param_6: float
    param_7: float
    param_8: float
    param_9: float
    param_10: float


class BatchMLRequest(BaseModel):
    objects: list[MLPredictionObject]


class BatchMLResponse(BaseModel):
    predictions: list[float]
