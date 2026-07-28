Z3 adapter boundary.

The Rust scheduler treats Z3 as an isolated backend process. Track 1 code owns
the protocol and health/cancellation contract here; solver-specific translation
will be expanded behind this adapter without leaking Z3-native types into UCIR.
