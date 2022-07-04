from siibra.retrieval.requests import HttpRequest, ZipfileRequest
import unittest
import pytest

from siibra import atlases, spaces
from siibra.core import Space
from pydantic import BaseModel


class TestSpaces(unittest.TestCase):

    space_id = "space_id"
    name = "space name"
    url = "space_url"
    ziptarget = "space_zip_target"
    template = "temp_file"
    ttype = "nii"

    json_space_with_zip = {
        "@id": "space1/minds/core/referencespace/v1.0.0",
        "name": name,
        "shortName": name,
        "templateType": ttype,
        "datasets": [
            {
                "@type": "fzj/tmp/volume_type/v0.0.1",
                "@id": "fzj/tmp/volume_type/v0.0.1/icbm152_2009c_nonlin_asym/nifti",
                "space_id": "space1/minds/core/referencespace/v1.0.0",
                "name": "icbm152_2009c_nonlin_asym/nifti",
                "volume_type": ttype,
                "url": url,
                "zipped_file": ziptarget,
            }
        ],
    }

    json_space_without_zip = {
        "@id": "space1/minds/core/referencespace/v1.0.0",
        "name": name,
        "shortName": name,
        "templateUrl": url,
        "templateFile": ziptarget,
        "templateType": ttype,
        "datasets": [
            {
                "@type": "fzj/tmp/volume_type/v0.0.1",
                "@id": "fzj/tmp/volume_type/v0.0.1/icbm152_2009c_nonlin_asym/nifti",
                "space_id": "space1/minds/core/referencespace/v1.0.0",
                "name": "icbm152_2009c_nonlin_asym/nifti",
                "volume_type": "nii",
                "url": "http://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/mni_icbm152_nlin_asym_09c_nifti.nii",
            }
        ],
    }

    def test_space_init(self):
        space = Space(self.space_id, self.name, self.url, self.ziptarget)
        self.assertIsNotNone(space)

    def test_space__from_json_with_zip(self):
        space = Space._from_json(self.json_space_with_zip)
        Space.REGISTRY.add(space.key, space)
        self.assertTrue(self.name in str(space))
        self.assertEqual(len(space.volumes), 1)
        vsrc = space.volumes[0]
        self.assertEqual(space.type, self.ttype)
        self.assertTrue(isinstance(vsrc._image_loader, ZipfileRequest))
        self.assertEqual(vsrc._image_loader.filename, self.ziptarget)

    def test_space__from_json_without_zip(self):
        space = Space._from_json(self.json_space_without_zip)
        Space.REGISTRY.add(space.key, space)
        self.assertTrue(self.name in str(space))
        self.assertEqual(len(space.volumes), 1)
        vsrc = space.volumes[0]
        self.assertTrue(isinstance(vsrc._image_loader, HttpRequest))

    def test_space_registry(self):
        spaces = atlases.MULTILEVEL_HUMAN_ATLAS.spaces
        self.assertEqual(len(spaces), 4)

all_spaces = [space for space in spaces]

@pytest.mark.parametrize('spc', all_spaces)
def test_json_serializable(spc: Space):
    assert issubclass(
        spc.to_model.__annotations__.get("return"),
        BaseModel,
    )
    import re
    model = spc.to_model()
    assert re.match(r"^[\w/\-.:]+$", model.id), f"model_id should only contain [\w/\-.:]+, but is instead {model.id}"
    import json
    json.loads(
        spc.to_model().json()
    )

if __name__ == "__main__":
    unittest.main()
