# Reversing with AtlasCTF

AtlasCTF treats reversing inputs as untrusted artifacts. Binary, source, and
trace frontends lower metadata into typed program models without executing
challenge code during intake. Later execution uses the Track 1 validation and
isolation contracts: read-only artifacts, no default networking, bounded
processes, and candidate replay before any `VALIDATED_SAT` result.

Current Track 2 capabilities:

- ELF metadata detection and x86-64 frontend boundary.
- Restricted C/Python checker intake with dangerous-token rejection.
- Concrete trace intake requiring explicit event payloads.
- Symbolic state, sparse memory, path predicates, concolic seed mutation,
  deterministic coverage scheduling, loop policies, and partial-on-budget
  semantics.
- Taint-guided slicing, constant branch pruning, loop-bound inference, and
  versioned pure-call summaries.
- Specialized GF(2), modular matrix, lattice-basis, and crypto recognizer
  strategy boundaries.

All final candidates must still pass UCIR validation and original checker replay
when available.
