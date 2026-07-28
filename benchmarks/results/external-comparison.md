# External solver comparison

Lower `mean_ns` is better. Atlas rows produce the bounded match stream up to the existing 1024-result cap; Z3 rows measure first-model solving through Python bindings.

| case | engine | mean ns | iterations | matches | notes |
|---|---:|---:|---:|---:|---|
| xor_width20 | atlas-native | 35 | 2000 | 1 | closed-form; evaluated=1 |
| add_width20 | atlas-native | 35 | 2000 | 1 | closed-form; evaluated=1 |
| checksum_width20 | atlas-native | 4320 | 20 | 1024 | closed-form; evaluated=1024 |
| rotxor_width24 | atlas-native | 36 | 2000 | 1 | closed-form; evaluated=1 |
| muladd_width24 | atlas-native | 55 | 2000 | 1 | closed-form; evaluated=1 |
| serial_bytes_width32 | atlas-native | 2881 | 200 | 256 | closed-form; evaluated=256 |
| xor_width20 | python-scalar | 90958366 | 3 | 1 | evaluated=1048576 |
| add_width20 | python-scalar | 76252900 | 3 | 1 | evaluated=1048576 |
| checksum_width20 | python-scalar | 58069166 | 3 | 1024 | evaluated=1048576 |
| rotxor_width24 | python-scalar | 2361359233 | 3 | 1 | evaluated=16777216 |
| muladd_width24 | python-scalar | 1662334066 | 3 | 1 | evaluated=16777216 |
| serial_bytes_width32 | python-scalar | 1046650100 | 3 | 1 | evaluated=16777216 |
| mod_sqrt_prime_101 | atlas-native-math | 4257 | 2000 | 2 |  |
| discrete_log_prime_29 | atlas-native-math | 4069 | 2000 | 1 |  |
| xor_width20 | z3-python-first-model | 794141 | 100 | 1 |  |
| add_width20 | z3-python-first-model | 224080 | 100 | 1 |  |
| checksum_width20 | z3-python-first-model | 3298789 | 100 | 1 |  |
| rotxor_width24 | z3-python-first-model | 452712 | 100 | 1 |  |
| muladd_width24 | z3-python-first-model | 211035 | 100 | 1 |  |
| serial_bytes_width32 | z3-python-first-model | 594458 | 100 | 1 |  |
| all | sage | n/a | n/a | n/a | sage CLI not found on PATH |
