# CTF benchmark results

Measured cases cover bounded-search kernels and exact math operations that show up in reversing, crypto, and serial/keygen CTF tasks.

| case | engine | mean ns | iterations | matches | CTF relevance |
|---|---:|---:|---:|---:|---|
| xor_width20 | atlas-native | 46 | 2000 | 1 | XOR masks and linear bit-vector checks common in crackmes and crypto warmups |
| add_width20 | atlas-native | 46 | 2000 | 1 | modular integer equality used in keygen and checksum gates |
| checksum_width20 | atlas-native | 5355 | 20 | 1024 | bounded checksum residue search for license and firmware puzzles |
| rotxor_width24 | atlas-native | 47 | 2000 | 1 | rotate/XOR mixing used in reversing and obfuscation challenges |
| muladd_width24 | atlas-native | 68 | 2000 | 1 | LCG-style modular arithmetic used in PRNG and serial checks |
| serial_bytes_width32 | atlas-native | 3416 | 200 | 256 | byte-constrained serial-prefix search |
| xor_width20 | python-scalar | 69641733 | 3 | 1 | XOR masks and linear bit-vector checks common in crackmes and crypto warmups |
| add_width20 | python-scalar | 60619900 | 3 | 1 | modular integer equality used in keygen and checksum gates |
| checksum_width20 | python-scalar | 50521500 | 3 | 1024 | bounded checksum residue search for license and firmware puzzles |
| rotxor_width24 | python-scalar | 1760885300 | 3 | 1 | rotate/XOR mixing used in reversing and obfuscation challenges |
| muladd_width24 | python-scalar | 1318953933 | 3 | 1 | LCG-style modular arithmetic used in PRNG and serial checks |
| serial_bytes_width32 | python-scalar | 793622666 | 3 | 1 | byte-constrained serial-prefix search |
| mod_sqrt_prime_101 | atlas-native-math | 4228 | 2000 | 2 | quadratic residue step used in CTF crypto |
| discrete_log_prime_29 | atlas-native-math | 4062 | 2000 | 1 | small finite-field discrete logarithm baseline |
| xor_width20 | z3-python-first-model | 670639 | 100 | 1 | XOR masks and linear bit-vector checks common in crackmes and crypto warmups |
| add_width20 | z3-python-first-model | 201715 | 100 | 1 | modular integer equality used in keygen and checksum gates |
| checksum_width20 | z3-python-first-model | 3068686 | 100 | 1 | bounded checksum residue search for license and firmware puzzles |
| rotxor_width24 | z3-python-first-model | 424869 | 100 | 1 | rotate/XOR mixing used in reversing and obfuscation challenges |
| muladd_width24 | z3-python-first-model | 210273 | 100 | 1 | LCG-style modular arithmetic used in PRNG and serial checks |
| serial_bytes_width32 | z3-python-first-model | 587959 | 100 | 1 | byte-constrained serial-prefix search |
