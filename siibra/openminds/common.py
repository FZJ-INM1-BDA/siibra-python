from pydantic.main import BaseModel


class CommonConfig:
    allow_population_by_field_name = True
    underscore_attrs_are_private = True

def get_openminds_dict(obj: BaseModel):
    def vocab_key(key: str):
        return key.replace('https://openminds.ebrains.eu/vocab/', '')
    vocabed_dict = { vocab_key(key): val for key, val in obj.dict(by_alias=True).items() }
    return {
        "@context": {
            "@vocab": "https://openminds.ebrains.eu/vocab/"
        },
        **vocabed_dict,
    }