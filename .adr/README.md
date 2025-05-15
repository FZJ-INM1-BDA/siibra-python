# Architecture Decision Record

> An architecture decision record (ADR) is a document that captures an important architectural decision made along with its context and consequences.[1]

## Overview

This document and the sibling documents aims to capture the:

- design philosophy of siibra-python and

- the architecture decisions that are codified in order to serve the design philosophies

This is done for the following purpose:

- Provide framework/guideline on how new functionalities should be implemented

- Codify context, stakeholders, and outcomes of previous architecture decisions

It should be noted that this document is for design philosophies and architectural decisions. Bug reports/feature requests, in most circumstances, do not belong in this module. 

## How to use

All documents in this directory can be edited via PR. They should be signed off by at least:

- two siibra-python maintainers, or
- a majority of siibra-python maintainers

whichever is lower. 

### Upsert documents

This section is relevant if you wish to insert (create new) or update (amend/deprecate) ANY documents in this directory.

1. (optional) Create an issue, tagged `[ADR proposal]`, outline the proposal.

2. Create a PR in the following format:

    - new ADRs must have filename must be in the format of `.adr/ADR{XXX}-{my-adr-title}.md` (replace `{XXX}` with the natural increment of the current last ADR; replace `{my-adr-title}` with an easy to read title)

    - updates to existing documents MUST must be documented in reverse chronological order in the changelog section (create one if none exist) at the bottom of the changed document

    - raise the PR to `main` branch

3. Ping at least one maintainer once the PR is ready to be reviewed

## References

[1] https://github.com/joelparkerhenderson/architecture-decision-record/tree/b5398157e6ae522ab7952f09815eeaf250454b36 (CC BY-NC-SA)