# ADR003: configuration and schema

## Status

Pending

## Context

siibra-python v1 relies on JSON files to load foundational datasets. This practice will be continued in v2, with some amendments. 

This document will both codify the existing design decisions, as well as request comments for the proposed update. They will be marked clearly.

## Proposal

1. (already adopted) Configuration used by siibra-python should reside in a separate repository[1]. This allows for the separation of concern between code (siibra-python) and data (siibra-configurations).

2. (already adopted) Released versions of siibra-python should reference a tagged version of siibra-configuration. This is in service of the reproducibility of the code.

    > n.b. this proposal is aimed mainly at stopping obvious foot-guns where pushing commits to a branch breaks past releases. This proposal will not help in the event of a/ force pushing the tags, b/ URLs referenced in the tagged configuration becomes stale.

3. (proposal) Synchronise release tags. At the moment, tags in siibra-configurations (`siibra-<version>` e.g. [2]) and siibra-python (`v<version>`, e.g. [3]) differ. For version 2, they should be synchronized: siibra-python version `v<version>` can expect to pull siibra-configuration with tag `v<version>`.

4. (proposal) json-schemas and validation of configuration JSONs should move to its own dedicated repository[2] (already adopted), and _both_ siibra-configuration and _siibra-python_ should treat the schema repository as the single source of truth on matters concerning schema/shape of JSON.

    > n.b. While in siibra-python v1, json-schema and schema validation for siibra-configurations JSONs existed and were used internally to ensure newly added configuration JSON files validated, they were not widely used due to the lack of discoverability, difficulty to setup. 

5. (proposal) pertain to 4., future discussions about schema should be at the schema repository[4]. This also includes addition of new schema.

6. (proposal) pertain to 4., old schemata are _never_ to be deleted, even if they become deprecated and obsolete. Schemata may be edited/updated, if the change is purely additive, and the added properties are not required (a.k.a., backwards compatible).

7. (proposal) pertain to 4., typing information should be automatically extracted (into `TypedDict` or `dataclass`)

## Affected Design Philosophies

### maintainability/debuggability (PRO)

Separating data-schema, data, and code can funnel discussions in the relevant repository, thereby improving maintainability.

Machine extracted typing information will prevent errors caused by typos.

### maintainability/debuggability (CON)

Separating the respository does increase the complexity and the number of repository maintainers will have to monitor.

## Stakeholders

- Ahmet Simsek
- Xiao Gui

## References


[1] https://github.com/FZJ-INM1-BDA/siibra-configurations

[2] https://github.com/FZJ-INM1-BDA/siibra-configurations/releases/tag/siibra-1.0.1-alpha.2

[3] https://github.com/FZJ-INM1-BDA/siibra-python/releases/tag/v1.0.1-alpha.2

[4] https://github.com/FZJ-INM1-BDA/siibra-schema