# External solver comparison

Lower `mean_ns` is better. Atlas rows produce the bounded match stream up to the existing 1024-result cap; Z3 rows measure first-model solving through Python bindings.

| case | engine | mean ns | iterations | matches | notes |
|---|---:|---:|---:|---:|---|
| xor_width20 | atlas-native | 46 | 2000 | 1 | closed-form; evaluated=1 |
| add_width20 | atlas-native | 46 | 2000 | 1 | closed-form; evaluated=1 |
| checksum_width20 | atlas-native | 5355 | 20 | 1024 | closed-form; evaluated=1024 |
| rotxor_width24 | atlas-native | 47 | 2000 | 1 | closed-form; evaluated=1 |
| muladd_width24 | atlas-native | 68 | 2000 | 1 | closed-form; evaluated=1 |
| serial_bytes_width32 | atlas-native | 3416 | 200 | 256 | closed-form; evaluated=256 |
| xor_width20 | python-scalar | 69641733 | 3 | 1 | evaluated=1048576 |
| add_width20 | python-scalar | 60619900 | 3 | 1 | evaluated=1048576 |
| checksum_width20 | python-scalar | 50521500 | 3 | 1024 | evaluated=1048576 |
| rotxor_width24 | python-scalar | 1760885300 | 3 | 1 | evaluated=16777216 |
| muladd_width24 | python-scalar | 1318953933 | 3 | 1 | evaluated=16777216 |
| serial_bytes_width32 | python-scalar | 793622666 | 3 | 1 | evaluated=16777216 |
| mod_sqrt_prime_101 | atlas-native-math | 4228 | 2000 | 2 |  |
| discrete_log_prime_29 | atlas-native-math | 4062 | 2000 | 1 |  |
| xor_width20 | z3-python-first-model | 670639 | 100 | 1 |  |
| add_width20 | z3-python-first-model | 201715 | 100 | 1 |  |
| checksum_width20 | z3-python-first-model | 3068686 | 100 | 1 |  |
| rotxor_width24 | z3-python-first-model | 424869 | 100 | 1 |  |
| muladd_width24 | z3-python-first-model | 210273 | 100 | 1 |  |
| serial_bytes_width32 | z3-python-first-model | 587959 | 100 | 1 |  |
| all | sage | n/a | n/a | n/a | sage CLI not found on PATH |
