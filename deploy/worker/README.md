Atlas worker deployment boundary.

Workers must run job containers as non-root, with networking disabled, read-only
artifact mounts, no Docker socket, and no host environment leakage. Coordinator
tests enforce these policy defaults at the contract layer.
