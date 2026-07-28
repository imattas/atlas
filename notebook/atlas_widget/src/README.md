# Atlas notebook widget UI shell

The production widget is driven exclusively by `EventEnvelopeV1` streams and
`SolveReportV1` exports. The checked-in Python model and Rust core define the
state contract consumed by graph, detail, progress, resource, provenance, replay,
and export views. Browser-specific adapters must not derive state from ad-hoc
solver internals.
