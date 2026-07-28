# External solver comparison

Lower `mean_ns` is better. Atlas rows produce the bounded match stream up to the existing 1024-result cap; Z3 rows measure first-model solving through Python bindings.

| case | engine | mean ns | iterations | matches | notes |
|---|---:|---:|---:|---:|---|
| xor_width20 | atlas-native | 64 | 2000 | 1 | closed-form; evaluated=1 |
| add_width20 | atlas-native | 38 | 2000 | 1 | closed-form; evaluated=1 |
| checksum_width20 | atlas-native | 6460 | 20 | 1024 | closed-form; evaluated=1024 |
| rotxor_width24 | atlas-native | 44 | 2000 | 1 | closed-form; evaluated=1 |
| muladd_width24 | atlas-native | 54 | 2000 | 1 | closed-form; evaluated=1 |
| serial_bytes_width32 | atlas-native | 3271 | 200 | 256 | closed-form; evaluated=256 |
| xor_width20 | python-scalar | 80995333 | 3 | 1 | evaluated=1048576 |
| add_width20 | python-scalar | 78207966 | 3 | 1 | evaluated=1048576 |
| checksum_width20 | python-scalar | 61269633 | 3 | 1024 | evaluated=1048576 |
| rotxor_width24 | python-scalar | 2389236500 | 3 | 1 | evaluated=16777216 |
| muladd_width24 | python-scalar | 1839236266 | 3 | 1 | evaluated=16777216 |
| serial_bytes_width32 | python-scalar | 1019182900 | 3 | 1 | evaluated=16777216 |
| xor_width20 | z3-python-first-model | 933292 | 100 | 1 |  |
| add_width20 | z3-python-first-model | 287174 | 100 | 1 |  |
| checksum_width20 | z3-python-first-model | 4169851 | 100 | 1 |  |
| rotxor_width24 | z3-python-first-model | 589055 | 100 | 1 |  |
| muladd_width24 | z3-python-first-model | 289176 | 100 | 1 |  |
| serial_bytes_width32 | z3-python-first-model | 816416 | 100 | 1 |  |
| all | sage | n/a | n/a | n/a | sage CLI not found on PATH |
