from pydantic import BaseModel, Field


class MLPredictionObject(BaseModel):
    f0: float = Field(alias='param_1')
    f1: float = Field(alias='param_2')
    f2: float = Field(alias='param_3')
    f3: float = Field(alias='param_4')
    f4: float = Field(alias='param_5')
    f5: float = Field(alias='param_6')
    f6: float = Field(alias='param_7')
    f7: float = Field(alias='param_8')
    f8: float = Field(alias='param_9')
    f9: float = Field(alias='param_10')

    class Config:
        allow_population_by_field_name: bool = True


class MLModelInput(BaseModel):
    objects: list[MLPredictionObject]


class MLModelOutput(BaseModel):
    predictions: list[float]
