# SparseIndex specification

This document describes the specification of a more memory efficient implementation of spatial index (at a slight cost to performance).

## Background

Each of the many statistical maps catalogued by siibra often contains hundreds of individual NifTI images. This pose a challenge to probabilistic assignments, where the operation would need to either

1/ read and cache all NifTI images into memory

2/ iterate over all NifTI images and only read them as needed with no caching

The former boasts significant improvement to speed to subsequent assignments, at the cost of significant memory usage[1]. Despite this, the performance of both approaches suffer from a cold start, as hundreds of NiFTi files need to be opened and potentially decompressed. (This performance penalty punishes the latter more severely, as each subsequent probabilistic assignment would incur performance cost over and over again.)

Spatial index exploits the nature of statistical maps in that they are often localized to a small number of voxels. Both memory usage and performance can be greatly improved by replacing the sparsely populated NiFTi images with a "spatial dictionary", where the voxel value encodes values on how the statistical value can be retrieved, from an "assignment table"

## Motivation of the current version of SparseIndex

The first implementation of spatial index (hereafter referred to as "old spatial index") contains several inefficiencies:

- spatial index encodes line number 

The old spatial index, each of the voxel encodes the line number in the corresponding assignment table. As a result, in order to decode an entry in the spatial dictionary, the "assignment table" file needs to be read from the beginning of the file.

- assignment table contains full region name

The old spatial index contains region names in the assignment table. This bloats the size of the assignment table. To alleviate the bloat, the assignment table is gzipped.

- the assignment table is stored in memory

The combination of both of the above means that the assignment table is often downloaded (or read) and decompressed in memory. 

- lack of version info

This makes upgrading/backwards compatibility difficult.

## Specification

SparseIndex contains four files, three of which are needed to decode the sparse index.

### 0/ Metadata file `{filename}`

utf-8 encoeded text file. *Must* start with `SPARSEINDEX-UTF8-V0`. Not needed to decode sparse index.

### 1/ Region decoding file `{filename}.sparseindex.alias.json`

utf-8 encoded json text file.

1.1/ *must* be decoded to a JSON object (e.g. not array).

1.2/ The JSON object *must* have string as keys, per JSON specification.

1.3/ The value of each key *must* be a JSON object with the following keys:

- `name`, the value must be string
- `bbox`, the value must be a list of int, with the size of exactly 6

### 2/ Spatial dictionary file `{filename}.sparseindex.voxel.nii.gz`

uint64 NiFTi 1 Image file. Each voxel encodes two 32 bit values, offset and bytes.

2.1/ Higher 32 bit (i.e. `voxel_value >> 32`) encodes offset. 

2.2/ Lower 32 bit (i.e. `voxel_value & 0xffffffff`) encodes bytes.

2.3/ For every voxel, offset (2.1) + bytes (2.2) *must* be less or equal to the file size of probability file (3.)

### 3/ Assignment table file `{filename}.sparseindex.probs.txt`

3.1/ For all voxel in (2.), byte range from offset (2.1) to 
offset + bytes (2.2) (i.e. `seek(offset); read(bytes)`) *must* be a utf-8 encoded string.

3.2/ All strings in 3.1. *must* be either an empty string (in the case `seek(offset); read(0)`) or a JSON object (i.e. not array etc).

3.3/ For each of the JSON object in 3.2, its keys *must* be found in the keys to the JSON object parsed in 1.2/


## Usage

Below demonstrates a step-by-step walkthrough on MESI's read and write implmentations. They have already been implemented in python, but is nevertheless useful for:

1/ translating it into other languages

2/ code auditing

### reading voxel index

To access the probability assignment at `[x, y, z]` voxel position

0/ read `{filename}.sparseindex.alias.json`, parse as JSON object.

1/ read the `uint64` value at the voxel position `[x, y, z]` from `{filename}.sparseindex.voxel.nii.gz`

2/ decode the `offset` by right shift 32 bits; `bytes` by using the bit mask `0xffffffff`

3/ read the probability file `{filename}.sparseindex.probs.txt`, seek `offset` and read `bytes`. 

4/ load result from 3/, decode as utf-8, parse the string as JSON. This should be a dictionary with `str` as key and `float` as value. 

5/ for each of the keys in 4/, use it as a string accessor on the JSON object retrieved in 0/ to retrieve a dictionary.

5.1/ Use `name` string accessor on the said dictionary to retrieve the region name.

5.2/ Use `bbox` string accessor on the said dictionary to retrieve the bounding box extrema


## Examples

### Writing

The below example writes Julich Brain 2.9 statistical map to base filename `icbm152_julich2_9` in directory `mesi`.

```python
import siibra
from tqdm import tqdm
from siibra.atlases.sparsemap import SparseIndex

mp = siibra.get_map("2.9", "icbm 152", "statistical")
spi = SparseIndex("icbm152_julich2_9", mode="w")


progress = tqdm(total=len(mp.regions), leave=True)
for regionname in mp.regions:
    volumes = mp.find_volumes(regionname)
    assert len(volumes) == 1
    volume = volumes[0]
    spi.add_img(volume.fetch(), regionname)
    progress.update(1)
progress.close()
spi.save()
```

### Reading

The below example reads the MESI saved above.

```python

import siibra
import numpy as np
from siibra.attributes.locations import Point
from siibra.atlases.sparsemap import SparseIndex


spi = siibra.atlases.sparsemap.SparseIndex("icbm152_julich2_9", mode="r")

pt_phys = [-4.077, -79.717, 11.356]
space = siibra.get_space("icbm 152")
pt = Point(coordinate=pt_phys, space_id=space.ID)
affine = np.linalg.inv(spi.affine)
pt_voxel = pt.transform(affine)
voxelcoord = np.array(pt_voxel.coordinate).astype("int")

val = spi.read([voxelcoord])
print(val) # prints [{'Area hOc2 (V2, 18) - left hemisphere': 0.33959856629371643, 'Area hOc1 (V1, 17, CalcS) - left hemisphere': 0.6118946075439453}]

```

If the spatial index is available over HTTP:

```python
import siibra

remote_spi = siibra.atlases.sparsemap.SparseIndex("https://data-proxy.ebrains.eu/api/v1/buckets/test-sept-22/icbm152_julich2_9.mesi", mode="r")

print(remote_spi.read([voxelcoord])) # prints [{'Area hOc2 (V2, 18) - left hemisphere': 0.33959856629371643, 'Area hOc1 (V1, 17, CalcS) - left hemisphere': 0.6118946075439453}]
```

## Advantages

- memory efficiency

    The only files that need to be stored in memory are `{filename}.sparseindex.voxel.nii.gz` and `{filename}.sparseindex.alias.json`. See [2] and [3] for a memory usage comparison.

- better cold start performance

    seek-read can be done without reading the entire file. If `{filename}.sparseindex.probs.txt` is available remotely, SparseIndex allows the probs value to be retrieved without downloading the entire file.

## Disadvantages

- easily invalidated

    As SparseIndex uses offset and byte ranges as pointers, any changes to the spatial index will likely invalidate the entire index. But generating index is a (relatively) quick process, and should not 

- mild performance penalty

    As `{filename}.sparseindex.probs.txt` will be on filesystem rather than in memory, repeated assignment can incur performance penalty. An escape hatch may be, allow the file content to be cached in memory, if a flag is set. 

- `{filename}.sparseindex.probs.txt` cannot be compressed (per file basis)
    
    As MESI uses byte range, no compression on the file level will work. However, storage is more readily available than memory. In most circumstances, this is a good tradeoff. 


## Potential future developments

- Binary Spec

Binary probability file/metadata file to improve performance and further reduce memory usage.

## References

[1] conservative estimate of memory usage of storing Julich Brain 3.0.3 high granularity statistical map (175 regions per hemisphere) in ICBM 152 nonlinear asymmetric space.

```
350 (number of niftis) * 193 * 229 * 193 (NifTI shape) * 4 (float32 datatype)

= 11942029400 bytes = 11389 mb
```

[2] conservative estimate of memory usage of MESI of a map in ICBM 152 nonlinear asymetric space

```
193 * 229 * 193 (NifTI shape) * 8 (uint64 datatype)

= 68240168 bytes = 65 mb
```

[3] conservative estimate of memory usage of old spatial index of a map in ICBM 152 nonlinear asymetric space

```
193 * 229 * 193 (NifTI shape) * 4 (uint32 datatype)

= 34120084 bytes = 32 mb

~1000 mb for `{filename}.sparseindex.probs.txt`

= ~1030 mb
```