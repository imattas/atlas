# AtlasCTF plugin conformance

Strategy plugins are described by manifests under `plugins/strategies/`.

Conforming plugins must:

- declare deterministic input and output contracts;
- preserve UCIR semantics and source provenance;
- fail closed on unsupported domains or widths;
- return candidates or explanations, not trusted facts;
- include tests or fixtures referenced by release evidence.

The built-in plugin set covers GF(2), modular matrix, lattice variants, and
crypto recognizers. Recognizers require independent confirmation before routing
specialized solvers.
