# MESI (More Efficient Spatial Index)

This document describes the specification of a more memory efficient implementation of spatial index (at a slight cost to performance).

## Background

Each of the many statistical maps catalogued by siibra often contains hundreds of individual NifTI images. This pose a challenge to probabilistic assignments, where the operation would need to either

1/ read and cache all NifTI images into memory

2/ iterate over all NifTI images and only read them as needed with no caching

The former boasts significant improvement to speed to subsequent assignments, at the cost of significant memory usage[1]. Despite this, the performance of both approaches suffer from a cold start, as hundreds of NiFTi files needs to be opened and potentially decompressed. (This performance penalty punishes the latter more severly, as each subsequent probabilistic assignment would incur performance cost over and over again.)

Spatial index exploits the nature of statistical maps in that they are often localized to a small number of voxels. Both memory usage and peroformance can be greatly improved by replacing the sparsely populated NiFTi images with a "spatial dictionary", where the voxel value encodes values on how the statistical value can be retrieved.

## Motiviation of MESI

The first implementation of spatial index (here after referred to as "old spatial index") contains several inefficiencies, some of these, MESI will try to address.

- spatial index encodes line number 

The old spatial index, each of the voxel encode the line number in the corresponding assignment table. As a result, in order to decode an entry in the spatial dictionary, one must start parsing from the beginning. Either read and parse the entire text, or until number of line breaks has been encountered.

- assignment table contains full region name

The old spatial index contain region names in the assignment table. This bloats the size of the assignment table. To alleviate the bloat, the assignment table is gzipped.

- the assignment table is stored in memory

The combination of both of above means that the assignment table is often downloaded (or read) and decompressed in memory. 

- lack of version info

This makes upgrading/backwards compatibility difficult.

## Specification

MESI contains three files, all of which are needed to decode the spatial index.

### 1/ Metadata file `{filename}.mesi.meta.txt`

utf-8 encoded text file, delineated by line breaks.

1.1/ First line *must* start with `MESI-UTF8-V0`.

1.2/ Each of the subsequent lines *must* be a valid JSON object (n.b.not array).

1.3/ Each of the JSON object referred to in 1.2 *must* contain the following keys:

- `regionname`, the value must be string
- `bbox`, the value must be a list of int, with the size of exactly 6

1.4/ (corollary to 1.3) There *must not* be a new line character at the end of file.

### 2/ Spatial dictionary file `{filename}.mesi.voxel.nii.gz`

uint64 NiFTi 1 Image file. Each voxel encodes two 32 bit values, offset and bytes.

2.1/ Higher 32 bit (i.e. `voxel_value >> 32`) encodes offset. 

2.2/ Lower 32 bit (i.e. `voxel_value & 0xffffffff`) encodes bytes.

2.3/ For every voxel, offset (2.1) + bytes (2.2) *must* be less or equal to the file size of probability file (3.)

### 3/ Probability file `{filename}.mesi.probs.txt`

3.1/ For all voxel in (2.), byte range from offset (2.1) to 
offset + bytes (2.2) (i.e. `seek(offset); read(bytes)`) *must* be a utf-8 encoded string.

3.2/ All string in 3.1. *must* be either an empty string or a JSON object.

3.3/ For all JSON object in 3.2, its keys *must* follow the following regex format: `^([1-9][0-9]+|0)$` (i.e. can be converted to `int`)

3.4/ Each of the keys in 3.3., on converted to `int` must be less than the size of number of JSON object parsed in (1.2.)


## Usage

Below demonstrates a step-by-step walkthrough on MESI's read and write implmementations. They have already been implemented in python, but is never the less useful for:

1/ translating it into other languages

2/ code auditing

### reading voxel index

To access the probability assignment at `[x, y, z]` voxel position

0/ read `{filename}.mesi.meta.txt`, ignore the first line, then split by new line characters. For each of the line as a JSON object.

1/ read the `uint64` value at the voxel position `[x, y, z]` from `{filename}.mesi.voxel.nii.gz`

2/ decode the `offset` by right shift 32 bits; `bytes` by using the bit mask `0xffffffff`

3/ read the probability file `{filename}.mesi.probs.txt`, seek `offset` and read `bytes`. 

4/ load result from 3/, decode as utf-8, parse the string as JSON. This should be a dictionary with `str` as key and `float` as value. 

5/ for each of the key in 4/, parse as `int`. Use the result value as the list index from the list in 0/ to retrieve a dictionary. Use `regionname` string accessor on the said dictionary to retrieve the region name


## Examples

### Writing

The below example writes Julich Brain 2.9 statistical map to base filename `icbm152_julich2_9` in directory `mesi`.

```python
import siibra
from tqdm import tqdm

mp = siibra.get_map("2.9", "icbm 152", "statistical")
spi = siibra.atlases.sparsemap.MESI("icbm152_julich2_9", "mesi", mode="w")


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
from tqdm import tqdm

spi = siibra.atlases.sparsemap.MESI("icbm152_julich2_9", "mesi", mode="r")

pt_phys = [-4.077, -79.717, 11.356]
space = siibra.get_space("icbm 152")
pt = Point(coordinate=pt_phys, space_id=space.ID)
affine = np.linalg.inv(spi.affine)
pt_voxel = pt.transform(affine)
voxelcoord = np.array(pt_voxel.coordinate).astype("int")

val = spi.read([voxelcoord])
print(val) # prints [{'Area hOc2 (V2, 18) - left hemisphere': 0.33959856629371643, 'Area hOc1 (V1, 17, CalcS) - left hemisphere': 0.6118946075439453}]
```

## Advantages

- memory efficiency

    The only files that needs to be read in memory is `{filename}.mesi.voxel.nii.gz` and `{filename}.mesi.meta.txt`. See [2] and [3] for a memory usage comparison.

- better cold start performance

    seek-read can be done without reading the entire file. If `{filename}.mesi.probs.txt` is available remotely, MESI allows the probs value to be retrieved without downloading the entire file.

## Disadvantages

- MESI much easily invalidated

    As MESI uses offset and byte ranges as pointer, any changes to the spatial index will likely invalidate the entire index. But generating index is a (relatively) quick process, and should not 

- mild performance penalty

    As `{filename}.mesi.probs.txt` will be on filesystem rather than in memory, repeated assignment can incur performance penalty. An escape hatch may be, allow the file content to be cached in memory, if a flag is set. 

- `{filename}.mesi.probs.txt` cannot be compressed (per file basis)
    
    As MESI uses byte range, no compression on the file level will work. However, storage is more readily available than memory. In most circumstance, this is a good tradeoff. 


## Potential future developments

- Binary Spec

Binary probability file/metadata file to improve performance and further reduce memory usage.

## References

[1] conservative estimate of memory usage of storing Julich Brain 3.0.3 high granularity statistical map (175 regions per hemisphere) in ICBM 152 nonlinear asymmetric space.

```
350 (number of niftis) * 193 * 229 * 193 (NifTI shape) * 4 (float32 datatype)

= 11942029400 bytes = 11389 mb
```

[2] conservative estimage of memory usage of MESI of a map in ICBM 152 nolinear assymetric space

```
193 * 229 * 193 (NifTI shape) * 8 (uint64 datatype)

= 68240168 bytes = 65 mb
```

[3] conservative estimage of memory usage of old spatial index of a map in ICBM 152 nolinear assymetric space

```
193 * 229 * 193 (NifTI shape) * 4 (uint32 datatype)

= 34120084 bytes = 32 mb

~1000 mb for `{filename}.sparseindex.probs.txt`

= ~1030 mb
```
