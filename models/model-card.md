# AtlasCTF Optional Ranker Model Card

The Track 4 ranker is optional and safety-restricted. Its output is limited to:

- ordered strategy identifiers
- budget multipliers
- explanation text

It cannot create facts, assumptions, candidates, validation evidence, trust
levels, or terminal result levels. If model metadata is missing or incompatible,
Atlas falls back to transparent rule-based ranking.

Training and test benchmark records are separated by the versioned benchmark
warehouse schema.
