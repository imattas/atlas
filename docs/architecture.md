# AtlasCTF architecture

Atlas is organized as a layered solve system:

1. Frontends safely lower source, binary, trace, and architecture-specific input
   into typed UCIR.
2. Analysis normalizes constraints, detects domains, slices programs, and emits
   provenance-preserving facts.
3. Planning chooses strategies, accelerators, and optional learned ordering
   without granting heuristic trust.
4. Execution runs local, SIMD/GPU, Z3, Sage, or distributed workers under
   cancellation and validation policies.
5. Reports and notebook replay consume versioned events and sanitized evidence.

The release manifest maps these layers to required suites and artifact evidence.
