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
        accelerator: None,
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

#[test]
fn indexes_device_validated_accelerator_evidence_by_backend() {
    let mut warehouse = BenchmarkWarehouse::new();
    let mut wgpu = record(CorpusSplit::Test);
    wgpu.accelerator = Some(atlas_benchmark_db::AcceleratorEvidence {
        backend: "WGPU".to_owned(),
        mode: "DeviceValidated".to_owned(),
        hardware: "WGPU device probe on Windows host".to_owned(),
    });
    warehouse.ingest(wgpu).unwrap();
    warehouse.ingest(record(CorpusSplit::Test)).unwrap();

    let records = warehouse.accelerator_backend("WGPU");

    assert_eq!(records.len(), 1);
    assert_eq!(
        records[0]
            .accelerator
            .as_ref()
            .map(|evidence| evidence.mode.as_str()),
        Some("DeviceValidated")
    );
    assert_eq!(warehouse.device_validated_accelerators().len(), 1);
}
