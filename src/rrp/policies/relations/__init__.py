"""Relation factors (D-144, docs/relations.md): one declarative registry of attention-interaction structure, the probes
that estimate it and the data that teaches it. `base` (token model, entries, specs, registry, resolution, hashing,
deploy guard), `ops` (operators, forms, FactorSite), `catalog` (the entries); the probe net is `rrp.policies.nets.probes.ReadoutProbe`."""
