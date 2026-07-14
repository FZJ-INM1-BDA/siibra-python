# Expansion with new content

`siibra-python` can load atlas content that is not included in its default
configuration. This guide explains how to:

* [1. Choose how to add your content](#1-choose-how-to-add-your-content)
* [2. Write a JSON specification](#2-write-a-json-specification)
* [3. Examples](#3-examples)
* [4. Test the specification](#4-test-the-specification)
* [5. Reuse specifications in a local configuration](#5-reuse-specifications-in-a-local-configuration)
* [6. Contribute content to the default configuration](#6-contribute-content-to-the-default-configuration)
* [7. When code changes are required](#7-when-code-changes-are-required)

For definitions of atlas elements, maps, spaces, data features, and anatomical
anchors, see {ref}`concepts`.

## 1. Choose how to add your content

Before writing a specification, decide how the content should be used.

```{mermaid}
flowchart TD
  supported["Is the object type supported by siibra-python?"]

  supported -- "no" --> code["A code change may be required"]
  code --> discuss["Open an issue or consult the developer documentation"]

  supported -- "yes" --> provider["Is the data format or provider supported?"]
  provider -- "no" --> code
  provider -- "yes" --> write["Write a JSON specification"]

  write --> test["Load and test it with siibra.from_json"]

  test -- "fails" --> fix["Check the schema, identifiers, and data source"]
  fix --> test

  test -- "works" --> reuse["How should the content be used?"]

  reuse -- "One-off local use" --> direct["Load the JSON directly"]
  reuse -- "Repeated local use" --> local["Add it to a local configuration"]
  reuse -- "Available by default" --> public["Prepare a contribution to siibra-configurations"]

  public --> shareable["Can the data and metadata be shared through a stable resource?"]
  shareable -- "yes" --> contribute["Open an issue or pull request"]
  shareable -- "no" --> local
```

The main routes are:

| Goal                                   | Recommended approach                          |
| -------------------------------------- | --------------------------------------------- |
| Test one object locally                | Load a JSON file with `siibra.from_json(...)` |
| Reuse several custom objects           | Create a local configuration                  |
| Add content to the default atlas       | Contribute to `siibra-configurations`         |
| Add an unsupported object or data type | Extend `siibra-python`                        |

Dynamic content discovered through live queries is implemented in
`siibra-python` and is not configured through static JSON files. See
{ref}`concepts` for the distinction between foundational, dynamic, and local
content.

## 2. Write a JSON specification

A **specification** is a JSON description of one siibra object.

A **configuration** is a collection of such specifications, usually organized
in a directory or repository.

A specification commonly defines:

1. the type and identity of the object,
2. metadata describing the object,
3. its anatomical or spatial references,
4. access to the underlying data.

The exact fields depend on the selected object type. Start from an existing
specification of the same type and validate the result against the corresponding
schema.

### Select the object type

The `@type` field determines which siibra object is created from the
specification.

For example:

```json
{
  "@type": "siibra/volume/v0.0.1"
}
```

Common configurable object categories include:

* atlases,
* parcellations,
* spaces,
* reference templates,
* labelled and statistical maps,
* data features,
* volumes and other data providers.

The object type must be supported by the installed version of
`siibra-python`.

When the intended object type is unclear, first identify the corresponding
concept on the {ref}`concepts` page and then inspect similar specifications in
the default configuration.

### Add identity and metadata

Most specifications require a stable identifier and a human-readable name.

A minimal metadata block may look like this:

```json
{
  "@id": "my-project/example-feature",
  "@type": "siibra/feature/volume_of_interest/v0.1",
  "name": "Example volume of interest"
}
```

Depending on the object type, additional metadata may include:

* modality,
* description,
* version,
* publications,
* dataset references,
* contributors,
* license information,
* links to external metadata records.

Use identifiers that are stable within the configuration. Avoid identifiers
that depend on local filenames or temporary directory structures.

### Add anatomical references

Atlas content should state what anatomy it refers to.

Depending on the object type, the specification may reference:

* a brain region,
* a parcellation,
* a reference space,
* a point or point cloud,
* a bounding box,
* an image extent,
* a combination of semantic and spatial information.

For example, an image feature may refer to a supported reference space:

```json
{
  "@id": "my-project/example-voi",
  "@type": "siibra/feature/volume_of_interest/v0.1",
  "name": "Example local volume of interest",
  "modality": "MRI",
  "space": {
    "name": "MNI 152 2009c nonlinear asymmetric"
  }
}
```

A feature associated with a known brain region may instead use a semantic
reference to that region.

Prefer the most precise anatomical information available. When both a semantic
region assignment and a spatial location are known, include both when supported
by the schema.

See {ref}`concepts` for the distinction between semantic and spatial anatomical
anchors.

### Define data providers

Providers describe how siibra retrieves the underlying data.

A provider entry consists of:

* a provider key describing the format or access mechanism,
* a path, URL, or provider-specific value.

A local NIfTI volume can be described schematically as:

```json
{
  "@type": "siibra/volume/v0.0.1",
  "providers": {
    "nii": "/path/to/local/image.nii.gz"
  }
}
```

A NIfTI file stored inside a local ZIP archive may be described as:

```json
{
  "@type": "siibra/volume/v0.0.1",
  "providers": {
    "zip/nii": "/path/to/archive.zip image_inside_archive.nii"
  }
}
```

Provider keys depend on the installed `siibra-python` version. Use an existing
map, template, or feature specification as the reference for the required
provider syntax.

The specification normally points to the data rather than embedding it.

## 3. Examples

The examples below illustrate the main structure of specifications. For a real
configuration, confirm all fields against the corresponding schema and a current
specification of the same type.

### Local image feature

This example defines an image feature in an existing reference space:

```json
{
  "@id": "my-project/example-voi",
  "@type": "siibra/feature/volume_of_interest/v0.1",
  "name": "Example local volume of interest",
  "modality": "MRI",
  "space": {
    "name": "MNI 152 2009c nonlinear asymmetric"
  },
  "providers": {
    "nii": "/path/to/project/data/example_voi.nii.gz"
  }
}
```

The specification contains:

* a feature type,
* a stable identifier,
* descriptive metadata,
* a spatial reference,
* a local data provider.

### Local labelled map

A labelled map additionally connects image labels to regions in a
parcellation:

```json
{
  "@id": "my-project/example-labelled-map",
  "@type": "siibra/map/labelled/v0.1",
  "name": "Example local labelled map",
  "space": {
    "@id": "example-space-id"
  },
  "parcellation": {
    "@id": "example-parcellation-id"
  },
  "volumes": [
    {
      "@type": "siibra/volume/v0.0.1",
      "providers": {
        "nii": "/path/to/project/data/example_labels.nii.gz"
      }
    }
  ],
  "indices": [
    {
      "region": "Example area 1",
      "label": 1
    },
    {
      "region": "Example area 2",
      "label": 2
    }
  ]
}
```

The referenced space, parcellation, and regions must exist in the active
configuration or be supplied together with the map.

Use a current labelled-map specification from `siibra-configurations` as the
starting point because required identifiers and index fields depend on the map
schema.

## 4. Test the specification

Test a new specification before adding it to a larger configuration.

A useful sequence is:

1. validate the JSON syntax and schema,
2. construct the object with `siibra.from_json(...)`,
3. inspect the resulting object,
4. fetch a small amount of data,
5. test the object in its intended workflow.

### Load it with `siibra.from_json`

The most direct functional test is:

```python
import siibra

obj = siibra.from_json("/path/to/my_specification.json")

print(type(obj))
print(obj)
```

This checks whether siibra can:

* parse the file,
* resolve the `@type`,
* construct the expected object,
* interpret its main metadata and references.

Loading the object does not necessarily fetch the underlying data.

### Fetch or inspect the data

After constructing the object, test its data access.

For image-like objects:

```python
data = obj.fetch()
print(data)
```

For a feature exposing tabular data:

```python
data = obj.data
print(data)
```

For large resources, begin with:

* a small local test file,
* a limited region of interest,
* a lower resolution,
* or another minimal representative sample.

Successful object construction alone does not confirm that provider paths,
remote resources, archive members, or data formats are valid.

### Validate against the schema

Configuration schemas define the expected fields for supported object types.

From a checkout containing the schema validation tools, a validation command
may look like:

```bash
python config_schema/check_schema.py /path/to/my_specification.json
```

Run the helper with `-h` when the available command-line options are unclear:

```bash
python config_schema/check_schema.py -h
```

Schema validation can detect:

* missing required fields,
* invalid field names,
* unsupported values,
* incorrect nested structures,
* incompatible schema versions.

It cannot confirm that referenced files or remote resources are accessible.
Always combine schema validation with a functional fetch test.

### Common problems

If `siibra.from_json(...)` fails, check:

* whether `@type` is supported,
* whether required fields are present,
* whether referenced identifiers exist,
* whether the schema version matches the installed package,
* whether file paths are correct,
* whether provider keys are supported,
* whether the object was placed in the correct configuration context.

If object creation succeeds but fetching fails, check:

* file or URL accessibility,
* archive member names,
* file format compatibility,
* permissions,
* provider-specific options,
* whether the resource contains the expected data.

## 5. Reuse specifications in a local configuration

Loading a JSON file directly is suitable for one-off tests. For repeated
workflows or several related objects, collect the specifications in a local
configuration.

A local configuration may mirror the structure of the default configuration:

```text
my-siibra-config/
├── atlases/
├── parcellations/
├── spaces/
├── maps/
└── features/
```

Only the required directories need to be present.

Place each specification in the directory corresponding to its object type.
Related objects should use consistent identifiers and references.

### Extend the default configuration

Use an extended configuration when custom content should be available together
with the default siibra content:

```python
import siibra

siibra.extend_configuration("/path/to/my-siibra-config")
```

This is appropriate when, for example:

* a custom map uses an existing siibra space,
* a local feature refers to a region from Julich-Brain,
* project-specific content should supplement the default atlas.

Identifiers in the local configuration must not unintentionally conflict with
existing identifiers.

### Replace the active configuration

Use a replacement configuration when the workflow should use only the supplied
configuration:

```python
import siibra

siibra.use_configuration("/path/to/my-siibra-config")
```

This is useful for:

* isolated tests,
* custom atlas deployments,
* configurations for other species,
* controlled environments with a fixed content set.

A replacement configuration must provide every atlas element required by the
workflow. References to objects that exist only in the default configuration
will otherwise remain unresolved.

## 6. Contribute content to the default configuration

Content intended for general use can be proposed for inclusion in the
`siibra-configurations` repository:

https://github.com/FZJ-INM1-BDA/siibra-configurations/tree/v1

Before preparing a large contribution, open an issue to discuss:

* whether the content fits the scope of the Multilevel Human Brain Atlas,
* which object type and schema should be used,
* where the data and metadata should be hosted,
* whether additional `siibra-python` functionality is required.

A contribution should normally include:

* a valid JSON specification in the appropriate directory,
* stable identifiers,
* clear names and descriptions,
* appropriate anatomical references,
* stable data-access paths,
* dataset and publication references where available,
* license and provenance information,
* a functional smoke test.

Data referenced by the default configuration should normally be available
through a stable and maintainable resource. Local filesystem paths are not
suitable for shared foundational content.

## 7. When code changes are required

A JSON specification can only create object types and use providers supported
by the installed `siibra-python` version.

Code changes may be required when introducing:

* a new feature modality,
* a new map or volume representation,
* an unsupported file format,
* a new provider or decoder,
* a new live query,
* behavior that cannot be expressed by an existing schema.

The implementation may involve:

1. defining or extending a Python class,
2. registering a configuration type,
3. adding a provider or decoder,
4. adding schema support,
5. adding tests,
6. adding an example specification,
7. documenting the new functionality.

These steps belong to the developer workflow rather than the normal
configuration workflow.

For implementation details, see {ref}`developer`.

# Further resources

* {ref}`concepts` explains the atlas and data-access concepts used by
  configuration specifications.
* {ref}`api` documents the Python objects used to load and inspect configured
  content.
* {ref}`developer` describes implementation details and extension points.
* {ref}`examples` provides executable workflows using atlas elements and data
  features.
* The default configuration is maintained in
  https://github.com/FZJ-INM1-BDA/siibra-configurations/.
* Schema definitions and validation tools are available in the configuration
  and `siibra-python` repositories.
