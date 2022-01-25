
from pydantic import BaseModel, Field


class SiibraAtIdModel(BaseModel):
    id: str = Field(..., alias="@id")

    def dict(self, *arg, **kwargs):
        kwargs["by_alias"] = True
        return super().dict(*arg, **kwargs)

    class Config:
        allow_population_by_field_name = True


class SiibraBaseModel(BaseModel):
    
    def sanitize_key(self, key: str):
        return key.replace("https://openminds.ebrains.eu/vocab/", "")

    def dict(self, *arg, **kwargs):
        kwargs["by_alias"] = True
        json_out = super().dict(*arg, **kwargs)
        return {
            "@context": {
                "@vocab": "https://openminds.ebrains.eu/vocab/"
            },
            **{ self.sanitize_key(key): value for key, value in json_out.items()}
        }
    class Config:
        allow_population_by_field_name = True
