from pydantic import BaseModel, Field


class ConfigBaseModel(BaseModel):

    def dict(self, *arg, **kwargs):
        kwargs["by_alias"] = True
        return super().dict(*arg, **kwargs)

    class Config:
        allow_population_by_field_name = True


class VocabModel(ConfigBaseModel):
    vocab: str = Field(..., alias="@vocab")


class SiibraAtIdModel(ConfigBaseModel):
    id: str = Field(..., alias="@id")


class SiibraBaseModel(ConfigBaseModel):
    context: VocabModel = Field(VocabModel(vocab="https://openminds.ebrains.eu/vocab/"), alias="@context")
    
