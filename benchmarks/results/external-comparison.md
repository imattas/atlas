# External solver comparison

Lower `mean_ns` is better. Atlas rows produce the bounded match stream up to the existing 1024-result cap; Z3 rows measure first-model solving through Python bindings.

| case | engine | mean ns | iterations | matches | notes |
|---|---:|---:|---:|---:|---|
| xor_width20 | atlas-native | 36 | 2000 | 1 | closed-form; evaluated=1 |
| add_width20 | atlas-native | 36 | 2000 | 1 | closed-form; evaluated=1 |
| checksum_width20 | atlas-native | 4050 | 20 | 1024 | closed-form; evaluated=1024 |
| rotxor_width24 | atlas-native | 37 | 2000 | 1 | closed-form; evaluated=1 |
| muladd_width24 | atlas-native | 53 | 2000 | 1 | closed-form; evaluated=1 |
| serial_bytes_width32 | atlas-native | 2884 | 200 | 256 | closed-form; evaluated=256 |
| xor_width20 | python-scalar | 70097300 | 3 | 1 | evaluated=1048576 |
| add_width20 | python-scalar | 64195966 | 3 | 1 | evaluated=1048576 |
| checksum_width20 | python-scalar | 51527066 | 3 | 1024 | evaluated=1048576 |
| rotxor_width24 | python-scalar | 1882198800 | 3 | 1 | evaluated=16777216 |
| muladd_width24 | python-scalar | 1306221500 | 3 | 1 | evaluated=16777216 |
| serial_bytes_width32 | python-scalar | 795405666 | 3 | 1 | evaluated=16777216 |
| mod_sqrt_prime_101 | atlas-native-math | 4293 | 2000 | 2 |  |
| discrete_log_prime_29 | atlas-native-math | 4056 | 2000 | 1 |  |
| xor_width20 | z3-python-first-model | 696308 | 100 | 1 |  |
| add_width20 | z3-python-first-model | 212843 | 100 | 1 |  |
| checksum_width20 | z3-python-first-model | 3122624 | 100 | 1 |  |
| rotxor_width24 | z3-python-first-model | 441355 | 100 | 1 |  |
| muladd_width24 | z3-python-first-model | 213245 | 100 | 1 |  |
| serial_bytes_width32 | z3-python-first-model | 599246 | 100 | 1 |  |
| all | sage | n/a | n/a | n/a | sage CLI not found on PATH |
