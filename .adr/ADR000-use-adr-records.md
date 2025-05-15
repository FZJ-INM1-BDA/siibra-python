# ADR000: Adopt version controlled ADR

## Status

Adopted

## Context

Architectural decisions made in in siibra-python is poorly documented, and are only "documented" via institutional knowledge, or worse yet, "some where in my inbox". 

This ADR proposes that all such decisions, including their context (stakeholders, context, specific examples - if any) in a version controlled fashion, and most importantly, in one place.

This gives siibra developers some guidelines when developing new features, updating current features.

## Proposal

All future architecture desicions MUST be recorded in the repository under `.adr/ADR{XXX}-{my-adr-title}.md`, where `{XXX}` is the next natural number to the last ADR accepted, and `{my-adr-title}` is the title of the proposal.

## Affected Design Philosophies

- maintainability (PRO)

Any architectural decisions made from this point onwards will be version controlled, enhancing transparency and 

## Stakeholders

- Ahmet Simsek
- Xiao Gui
