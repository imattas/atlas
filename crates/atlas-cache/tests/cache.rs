//! Cache integrity tests.

use atlas_cache::{Cache, CacheError};

#[test]
fn cache_round_trips_verified_bytes() {
    let mut cache = Cache::new();
    cache.put("graph", b"bytes".to_vec());

    assert_eq!(cache.get("graph").unwrap(), Some(b"bytes".to_vec()));
}

#[test]
fn cache_detects_corrupt_entries() {
    let mut cache = Cache::new();
    cache.put("graph", b"bytes".to_vec());
    cache.corrupt_for_test("graph");

    assert_eq!(cache.get("graph"), Err(CacheError::Corrupt));
}
