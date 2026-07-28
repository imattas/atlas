# External solver comparison

Lower `mean_ns` is better. Atlas rows produce the bounded match stream up to the existing 1024-result cap; Z3 rows measure first-model solving through Python bindings.

| case | engine | mean ns | iterations | matches | notes |
|---|---:|---:|---:|---:|---|
| xor_width20 | atlas-native | 49 | 2000 | 1 | closed-form; evaluated=1 |
| add_width20 | atlas-native | 41 | 2000 | 1 | closed-form; evaluated=1 |
| checksum_width20 | atlas-native | 9475 | 20 | 1024 | closed-form; evaluated=1024 |
| rotxor_width24 | atlas-native | 69 | 2000 | 1 | closed-form; evaluated=1 |
| muladd_width24 | atlas-native | 94 | 2000 | 1 | closed-form; evaluated=1 |
| serial_bytes_width32 | atlas-native | 4364 | 200 | 256 | closed-form; evaluated=256 |
| xor_width20 | python-scalar | 102930666 | 3 | 1 | evaluated=1048576 |
| add_width20 | python-scalar | 94149666 | 3 | 1 | evaluated=1048576 |
| checksum_width20 | python-scalar | 77974600 | 3 | 1024 | evaluated=1048576 |
| rotxor_width24 | python-scalar | 2624498400 | 3 | 1 | evaluated=16777216 |
| muladd_width24 | python-scalar | 1780682433 | 3 | 1 | evaluated=16777216 |
| serial_bytes_width32 | python-scalar | 1142056666 | 3 | 1 | evaluated=16777216 |
| xor_width20 | z3-python-first-model | 963754 | 100 | 1 |  |
| add_width20 | z3-python-first-model | 304316 | 100 | 1 |  |
| checksum_width20 | z3-python-first-model | 4189285 | 100 | 1 |  |
| rotxor_width24 | z3-python-first-model | 635916 | 100 | 1 |  |
| muladd_width24 | z3-python-first-model | 306277 | 100 | 1 |  |
| serial_bytes_width32 | z3-python-first-model | 842667 | 100 | 1 |  |
| all | sage | n/a | n/a | n/a | sage CLI not found on PATH |
