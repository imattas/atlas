//! Benchmark warehouse tests.

use std::collections::BTreeMap;

use atlas_benchmark_db::{
    BenchmarkError, BenchmarkRecord, BenchmarkWarehouse, CorpusSplit, BENCHMARK_SCHEMA_MAJOR,
};

fn record(split: CorpusSplit) -> BenchmarkRecord {
    BenchmarkRecord {
        schema_major: BENCHMARK_SCHEMA_MAJOR,
        challenge_id: "xor-1".to_owned(),
        strategy_id: "gf2".to_owned(),
        features: BTreeMap::from([("vars".to_owned(), 32.0)]),
        runtime_ms: 10,
        split,
    }
}

#[test]
fn ingests_schema_and_keeps_train_test_separate() {
    let mut warehouse = BenchmarkWarehouse::new();
    warehouse.ingest(record(CorpusSplit::Train)).unwrap();
    warehouse.ingest(record(CorpusSplit::Test)).unwrap();

    assert_eq!(warehouse.split(CorpusSplit::Train).len(), 1);
    assert_eq!(warehouse.split(CorpusSplit::Test).len(), 1);
}

#[test]
fn rejects_incompatible_schema_and_missing_features() {
    let mut warehouse = BenchmarkWarehouse::new();
    let mut bad = record(CorpusSplit::Train);
    bad.schema_major = 2;
    assert_eq!(
        warehouse.ingest(bad),
        Err(BenchmarkError::IncompatibleSchema)
    );

    let mut missing = record(CorpusSplit::Train);
    missing.features.clear();
    assert_eq!(
        warehouse.ingest(missing),
        Err(BenchmarkError::MissingField("features".to_owned()))
    );
}
