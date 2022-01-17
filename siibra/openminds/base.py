from pydantic import BaseModel

class SiibraBaseModel(BaseModel):
    class Config:
        allow_population_by_field_name = True