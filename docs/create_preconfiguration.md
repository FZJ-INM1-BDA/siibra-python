# Working with your data


## Overview

`siibra` uses json files to preconfigure data, such as parcellation maps, reference spaces,
and data features. This guide walks you through how to add your own data whether it
is public or private.

You can now use this json to load your desired data as a siibra object and take
advantage its functionalities. You can also share this json with your colleagues
to collabrate.

At this stage, if you'd like to see your data in siibra-toolbox as default,
please contact the siibra team. (Most configurations are also automatically
added to ``siibra-explorer``.)


## Determine configuration type

`siibra` accepts a variety of configuration types. Nonetheless, it has limitations. 

### 1. Atlas concepts
- Atlas: 
- Parcellation:
- Reference space:
- Map: 
  - Labelled map: A volume describing a parcellation map such that integer labels determine areas on a reference space.
  - Statistical map: A set of volumes with probability (or other statistical values) for a subset of areas from a parcellation.

### 2. Data features
A variety of data features can be preconfigured. However, while some data types are very similar, the modality has to
be supported by a specific siibra-python version (this will not be required with version 2).

- Parcellation-based connectivity: Connectivity matrix calculated for a specific parcellation.
  - Functional connectivity
  - Streamline counts
  - Streamline lengths
  - Anatomo functional connectivity (Functional Tractography)
  - Tracing connectivity
- Image sections: 2D image sections in the format of nifti and neuroglancer/precomputed.
  - Cellbody stained section
- Volume of interest
  - Cellbody stained volume reconstructions
  - Blockface
  - Diffusion tensor imaging (DTI)
  - HSV fibre orientation map
  - Transmittance
  - Magnetic resonance imaging (MRI)
  - Phase Contrast X-ray Tomography (XPCT)
  - Light Sheet Fluorescence Microscopy (LSFM)
- Tabular
  - Receptor density fingerprint
  - Receptor density profile
  - Parcellation-based activity timeseries
    - Parcellation-based BOLD (blood-oxygen-level-dependent) signals

#### Anatomical anchor
An important part of data features is how they are linked to atlas concepts. A data feature can be linked to
1. A semantic area (region, parcellation),
2. A spatial location (bounding box, point cloud),
or both. This is crucial for `siibra` to assign this feature to other atlas concepts.

#### Integration of additional modalities

1. determine a "@type" and add this to the newly created configuration json.
2. Add the configuration into siibra-configurations by "features/<data_type>/<other_categoryies_if_any><new_modality_name>"
3. Declare a new class and ensure `configuration_folder="features/<data_type>/<other_categoryies_if_any><new_modality_name>"` is passed with the class declaration.
4. Export the newly created class at the appropriate module. 
5. Then, update `siibra.configuration.factory.py` to digest type defined in step 1.

To test:
Use `siibra.use_configuration(<path_to_local_config>)` (or `siibra.extend_configuration(<path_to_local_config>)`) to allow siibra to find this json path_to_local_config would be a clone of https://github.com/FZJ-INM1-BDA/siibra-configurations.

#### Supported data formats

``siibra.retrieval.requests.DECODERS`` lists all the file types that siibra can directly digest. Note that you can specify csv decoders in further detail, see connectivity configurations for examples.

If there is volume data, make sure to write the correct provider and see if the image or mesh type is supported by siibra, see ``siibra.volumes.volume.SUPPORTED_FORMATS``. Example:
```json
  "volumes": [
		{
			"@type": "siibra/volume/v0.0.1",
			"providers": {
				"zip/nii": "http://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/mni_icbm152_nlin_asym_09c_nifti.zip mni_icbm152_t1_tal_nlin_asym_09c.nii"
			}
		}
	]
```

## Create your json configuration

If the modality exists, it helps to check existing configurations for examples.
In general, 

1. a unique and descriptive json file name
2. a valid data destination:
  - a publicly accessible URL or
  - Local files can be accessed via a [python web server](https://docs.python.org/3/library/http.server.html) such that the file can be served over the local host, e.g. http://localhost:8000/<path_to_file>. This is particularly useful for sensitive data.


## Testing
1. Json schemas are located in ``config_schema`` folder within `siibra-python <https://github.com/FZJ-INM1-BDA/siibra-python>`. These are meant to be a guide, particularly for the minimum requirements. You can test (given that configuration type is known to a specific siibra-python version) if a json was formed according to the requirements by
```python
python <path_to_siibra_python>/config_schema/check_schema.py <path_to_siibra_config>
```
2. Test if siibra can digest the configuration:
```python
obj = siibra.from_json("path_to_json")
```
This is a good first check to see if the object type is achieved (``print(type(obj))``), check the properties of ``obj`` such as `data`, fetch the volumes configured by `obj.fetch()`.

If there are any issues, trace back your steps and see if everything is in order. If you have questions, please see FAQ on how to reach siibra team.
