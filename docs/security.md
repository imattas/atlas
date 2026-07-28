# AtlasCTF security model

Atlas separates untrusted inputs, heuristic analysis, solver results, and
validated outcomes.

- Heuristics may propose ordering, budgets, features, or assumptions, but cannot
  create trusted facts or terminal results.
- Candidate models are promoted only after UCIR evaluation and checker
  validation.
- Distributed workers are isolated, authenticated, replay-protected, and
  scheduled with least-capability matching.
- Exact rational, modular-linear, polynomial, and bit-vector math routes execute
  through Atlas-owned native code rather than external SMT/algebra backends.
- Reports and notebook views apply explicit redaction before display or export.
- Binary/source intake rejects malformed, oversized, and unsafe artifacts before
  lowering.

Security-sensitive changes must run the full verification profile and update the
release manifest evidence when capabilities change.
