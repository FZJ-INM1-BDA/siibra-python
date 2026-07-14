# Working with limited or no network access

`siibra-python` normally retrieves atlas metadata and data from distributed
online resources when they are first requested. Retrieved files are stored in a
local cache and can be reused by later operations.

This makes it possible to prepare many workflows for environments with limited
or unavailable network access. However, offline support is not complete. Some
operations depend on remote services, and very large tiled image resources
cannot currently be prepared automatically for offline use.

This guide explains:

* [1. What can work offline](#1-what-can-work-offline)
* [2. How to prepare the local cache](#2-how-to-prepare-the-local-cache)
* [3. How `siibra.warm_cache` works](#3-how-siibrawarm_cache-works)
* [4. Limitations for tiled image resources](#4-limitations-for-tiled-image-resources)
* [5. Operations that require a network connection](#5-operations-that-require-a-network-connection)
* [6. How to verify an offline workflow](#6-how-to-verify-an-offline-workflow)

## 1. What can work offline

Whether a workflow works without network access depends on which atlas
elements, data files, and services it uses.

| Operation                                           | Offline use            |
| --------------------------------------------------- | ---------------------- |
| Inspecting already loaded atlas metadata            | Possible               |
| Using local configuration files                     | Possible               |
| Reading local data providers                        | Possible               |
| Accessing files already present in the siibra cache | Possible               |
| Accessing an uncached remote file                   | Not possible           |
| Fetching arbitrary data from tiled image resources  | Not generally possible |
| Transforming locations between reference spaces     | Not possible           |
| Running live queries against external services      | Not possible           |

A workflow is suitable for offline use when all required configuration and data
are locally available and the workflow does not invoke a remote service.

For example, a workflow that loads a cached MNI template and performs an
analysis within MNI space may work offline. A workflow that subsequently
transforms a coordinate into BigBrain space will still require a network
connection.

```{important}
Successful access to an atlas object does not mean that all of its data are
available locally. siibra loads data lazily, so a later call to `fetch()` may
still require a download.
```

## 2. How to prepare the local cache

The safest way to prepare a workflow is to run the actual workflow once while a
network connection is available.

During this preparation run:

1. load the required atlas elements,
2. fetch all required templates, maps, and features,
3. perform representative assignments and data-access operations,
4. confirm that all expected outputs are produced,
5. repeat the workflow without network access.

This approach downloads only the content that the workflow actually uses.

For example:

```python
import siibra

space = siibra.spaces.get("mni152")
template = space.get_template()
template_image = template.fetch()

parcellation_map = siibra.get_map(
    parcellation="julich",
    space=space,
    maptype="statistical",
)

map_image = parcellation_map.fetch()
```

After the requested resources have been fetched, later calls can reuse their
cached copies.

### Increase the cache size when necessary

siibra maintains the local cache below a configured size. The default limit is
relatively small compared with the combined size of atlas maps, templates, and
features.

Increase the limit before downloading a larger collection:

```python
import siibra

siibra.set_cache_size(20)
```

The value is specified in gigabytes.

Choose a limit that is large enough for the intended workflow. Otherwise,
previously downloaded resources may be removed during cache maintenance as new
files are added.

## 3. How `siibra.warm_cache` works

`siibra.warm_cache` runs registered preparation functions before the
corresponding objects or data are requested by a workflow.

It supports different warm-up levels.

### Preload configured instances

Calling `warm_cache()` without an explicit level uses
`WarmupLevel.INSTANCE`:

```python
import siibra

siibra.warm_cache()
```

This constructs preconfigured atlas concepts such as spaces, parcellations, and
features in advance.

It is useful for:

* resolving configuration problems early,
* preparing configured objects before running a workflow,
* avoiding incremental object construction during later operations.

It does **not** mean that all referenced map, template, and feature data have
been downloaded.

### Preload registered data

To also run data-level warm-up functions, use `WarmupLevel.DATA`:

```python
import siibra

siibra.set_cache_size(20)
siibra.warm_cache(level=siibra.WarmupLevel.DATA)
```

This requests the data resources that have registered a data-level warm-up
operation and stores retrieved files in the local cache.

The required download size depends on the active configuration and the
available content. Set an appropriate cache limit before starting the operation.

```{note}
`WarmupLevel.DATA` does not guarantee that every possible resource used by
siibra is available offline. It only runs the data warm-up operations
implemented by the corresponding object and provider types.
```

For a focused analysis, running the actual workflow online once is usually more
predictable and requires less storage than warming all registered data.

## 4. Limitations for tiled image resources

Some high-resolution images are stored as multi-resolution tiled resources
rather than as individual downloadable files.

Examples include microscopic whole-brain images for which siibra retrieves only
the tiles required for a selected resolution and region of interest.

Preparing a complete tiled resource for offline use can require very large
amounts of storage. It may involve downloading a substantial number of chunks
across different resolutions.

Automatic offline preparation of tiled image resources is not currently
implemented in `siibra.warm_cache`.

As a result:

* warming the cache does not create a complete local copy of tiled images;
* fetching a new region or resolution may still require network access;
* workflows using high-resolution cloud images should not be assumed to work
  offline;
* previously fetched files do not necessarily cover later requests for other
  regions or resolutions.

A future implementation would need to define which resolutions and spatial
extents should be downloaded, estimate their required storage, and store the
result in a form usable by the corresponding tiled-image provider.

## 5. Operations that require a network connection

### Spatial transformations

Nonlinear transformations between supported reference coordinate systems are
currently performed through a remote spatial-transformation service.

For example:

```python
point_mni = siibra.Point(
    (27.75, -32.0, 63.725),
    space="mni152",
)

point_bigbrain = point_mni.warp("bigbrain")
```

The call to `warp()` requires access to the transformation service. Warming the
data cache does not make this service available locally.

Spatial transformations are therefore not available in a fully offline
workflow.

See {ref}`spatial transformations <spatial-transformations>` for the conceptual
distinction between locations, spaces, and cross-space transformations.

Affine transformations supplied and applied directly by the user are different:
they can be computed locally, but siibra cannot verify that a user-provided
affine correctly relates the declared source and target spaces.

### Live queries

Live queries discover or construct data features by contacting external
services at runtime.

Examples include queries to:

* the EBRAINS Knowledge Graph,
* the Allen Human Brain Atlas,
* cloud-hosted image resources,
* other service-specific data providers.

Because the result is obtained from the external service, a live query cannot
run without network access.

Cached files from an earlier query do not necessarily replace the query itself.
The service may still be needed to discover matching features and retrieve
their current metadata.

For offline workflows, prefer:

* foundational features defined by the active configuration,
* data that have already been fetched and cached,
* local features loaded from JSON specifications,
* local files with explicit anatomical references.

See {ref}`concepts` for the distinction between foundational content, local
content, and dynamic content from live queries.

## 6. How to verify an offline workflow

Test the complete workflow before moving it into a restricted environment.

A practical sequence is:

1. install the required `siibra-python` version;
2. activate the intended configuration;
3. increase the cache size when necessary;
4. call `siibra.warm_cache` at the appropriate level;
5. run the complete analysis once with network access;
6. disable network access;
7. restart Python to avoid relying on objects held only in memory;
8. run the analysis again from beginning to end.

Restarting Python is important because `WarmupLevel.INSTANCE` also preloads
objects into memory. A new process provides a more realistic test of whether
the required configuration and data are actually available locally.

Pay particular attention to operations that may implicitly require network
access, including:

* fetching a map or template for the first time,
* retrieving a data feature,
* accessing another part of a tiled image,
* transforming a location into another reference space,
* performing a feature query backed by a live query.

```{warning}
A workflow that succeeds once while online is not automatically ready for
offline use. Verify it in a new Python process with the network disabled.
```

# Further resources

* {ref}`concepts` explains lazy data access, caching, live queries, and spatial
  transformations.
* {ref}`api` documents `siibra.warm_cache`, `siibra.WarmupLevel`, and
  `siibra.set_cache_size`.
* {ref}`Configuring custom atlas content <configuring-atlas-content>` explains
  how to use local specifications and local configurations.
* {ref}`examples` provides executable workflows that can be used to identify
  which resources an analysis requires.
