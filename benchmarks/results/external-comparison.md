# External solver comparison

Lower `mean_ns` is better. Atlas rows produce the bounded match stream up to the existing 1024-result cap; Z3 rows measure first-model solving through Python bindings.

| case | engine | mean ns | iterations | matches | notes |
|---|---:|---:|---:|---:|---|
| xor_width20 | atlas-native | 35 | 2000 | 1 | closed-form; evaluated=1 |
| add_width20 | atlas-native | 51 | 2000 | 1 | closed-form; evaluated=1 |
| checksum_width20 | atlas-native | 6395 | 20 | 1024 | closed-form; evaluated=1024 |
| xor_width20 | python-scalar | 88663000 | 3 | 1 |  |
| add_width20 | python-scalar | 79692000 | 3 | 1 |  |
| checksum_width20 | python-scalar | 62699933 | 3 | 1024 |  |
| xor_width20 | z3-python-first-model | 892567 | 100 | 1 |  |
| add_width20 | z3-python-first-model | 297418 | 100 | 1 |  |
| checksum_width20 | z3-python-first-model | 3958632 | 100 | 1 |  |
| all | sage | n/a | n/a | n/a | sage CLI not found on PATH |
