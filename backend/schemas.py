"""Request schema shared by the prediction API and the online inference client."""

from pydantic import BaseModel, Field


class Features(BaseModel):
    value: float = Field(ge=0)
    lag_1h: float = Field(ge=0)
    lag_2h: float = Field(ge=0)
    lag_24h: float = Field(ge=0)
    rolling_mean_3h: float = Field(ge=0)
    rolling_mean_24h: float = Field(ge=0)
    hour: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)
    month: int = Field(ge=1, le=12)
    is_weekend: int = Field(ge=0, le=1)
