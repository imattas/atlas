//! Coordinator/worker integration tests.

use atlas_coordinator::{
    caps, ArtifactEnvelope, Coordinator, CoordinatorError, JobEnvelope, SqliteLeaseStore,
    WorkerResult,
};
use atlas_worker::{SandboxControl, SandboxPolicy, WorkerRegistration};
use std::path::PathBuf;

fn registration(id: &str, cert: &str, labels: &[&str]) -> WorkerRegistration {
    WorkerRegistration {
        worker_id: id.to_owned(),
        certificate_fingerprint: cert.to_owned(),
        capabilities: caps(labels),
    }
}

fn temp_db_path(prefix: &str) -> PathBuf {
    std::env::temp_dir().join(format!("{prefix}-{}.sqlite", std::process::id()))
}

#[test]
fn valid_registration_and_untrusted_certificate_are_handled() {
    let mut coordinator = Coordinator::new(["trusted".to_owned()]);

    assert!(coordinator
        .register(registration("worker-1", "trusted", &["cpu"]))
        .is_ok());
    assert_eq!(
        coordinator.register(registration("worker-2", "bad", &["cpu"])),
        Err(CoordinatorError::UntrustedCertificate)
    );
}

#[test]
fn scheduling_uses_least_capable_matching_worker() {
    let mut coordinator = Coordinator::new(["trusted".to_owned()]);
    coordinator
        .register(registration(
            "fat",
            "trusted",
            &["cpu", "gpu", "native-math"],
        ))
        .unwrap();
    coordinator
        .register(registration("thin", "trusted", &["cpu", "gpu"]))
        .unwrap();
    let job = JobEnvelope::new("job", "hash", ["gpu".to_owned()], "secret", 10);

    assert_eq!(coordinator.schedule(&job).unwrap(), "thin");
}

#[test]
fn disconnect_requeues_leased_jobs_to_remaining_capable_workers() {
    let mut coordinator = Coordinator::new(["trusted".to_owned()]);
    coordinator
        .register(registration("worker-1", "trusted", &["cpu"]))
        .unwrap();
    coordinator
        .register(registration("worker-2", "trusted", &["cpu"]))
        .unwrap();
    let job = JobEnvelope::new("job", "hash", ["cpu".to_owned()], "secret", 10);

    assert_eq!(coordinator.lease_job(&job).unwrap(), "worker-1");
    assert_eq!(coordinator.active_lease("job"), Some("worker-1"));
    assert_eq!(
        coordinator.worker_disconnected("worker-1"),
        vec!["job".to_owned()]
    );
    assert_eq!(coordinator.active_lease("job"), None);
    assert_eq!(coordinator.lease_job(&job).unwrap(), "worker-2");
}

#[test]
fn coordinator_snapshot_restores_workers_cancellations_leases_and_results() {
    let mut coordinator = Coordinator::new(["trusted".to_owned()]);
    coordinator
        .register(registration("worker-1", "trusted", &["cpu"]))
        .unwrap();
    let job = JobEnvelope::new("job", "hash", ["cpu".to_owned()], "secret", 10);
    let duplicate_job = JobEnvelope::new("duplicate", "hash", ["cpu".to_owned()], "secret", 10);
    let duplicate_result = WorkerResult::new("duplicate", "result", "secret");

    assert_eq!(coordinator.lease_job(&job).unwrap(), "worker-1");
    coordinator.cancel("job");
    coordinator
        .submit_result(&duplicate_job, &duplicate_result, "secret", 1)
        .unwrap();

    let mut restored = Coordinator::restore(coordinator.snapshot());

    assert_eq!(restored.active_lease("job"), Some("worker-1"));
    assert_eq!(restored.schedule(&job).unwrap(), "worker-1");
    assert!(restored.is_cancelled("job"));
    assert_eq!(
        restored.submit_result(&duplicate_job, &duplicate_result, "secret", 1),
        Err(CoordinatorError::DuplicateResult)
    );
}

#[test]
fn sqlite_lease_store_recovers_leases_and_result_dedup_after_restart() {
    let db_path = temp_db_path("atlas-coordinator-lease-store");
    let mut coordinator = Coordinator::new(["trusted".to_owned()]);
    coordinator
        .register(registration("worker-1", "trusted", &["cpu"]))
        .unwrap();
    let leased = JobEnvelope::new("leased", "hash", ["cpu".to_owned()], "secret", 20);
    let completed = JobEnvelope::new("completed", "hash", ["cpu".to_owned()], "secret", 20);
    let completed_result = WorkerResult::new("completed", "result", "secret");

    coordinator.lease_job(&leased).unwrap();
    coordinator.cancel("leased");
    coordinator
        .submit_result(&completed, &completed_result, "secret", 1)
        .unwrap();

    let store = SqliteLeaseStore::open(&db_path).unwrap();
    store.save_snapshot(&coordinator.snapshot()).unwrap();
    let mut restored = Coordinator::restore(store.load_snapshot().unwrap());

    assert_eq!(restored.active_lease("leased"), Some("worker-1"));
    assert!(restored.is_cancelled("leased"));
    assert_eq!(
        restored.submit_result(&completed, &completed_result, "secret", 1),
        Err(CoordinatorError::DuplicateResult)
    );

    let _ = std::fs::remove_file(db_path);
}

#[test]
fn worker_heartbeat_updates_capabilities_and_records_liveness_tick() {
    let mut coordinator = Coordinator::new(["trusted".to_owned()]);
    coordinator
        .register(registration("worker-1", "trusted", &["cpu"]))
        .unwrap();
    let gpu_job = JobEnvelope::new("job", "hash", ["gpu".to_owned()], "secret", 10);

    assert_eq!(
        coordinator.schedule(&gpu_job),
        Err(CoordinatorError::NoCapableWorker)
    );

    coordinator
        .heartbeat("worker-1", caps(&["cpu", "gpu"]), 42)
        .unwrap();

    assert_eq!(coordinator.last_heartbeat("worker-1"), Some(42));
    assert_eq!(coordinator.schedule(&gpu_job).unwrap(), "worker-1");
}

#[test]
fn artifact_fetch_is_content_addressed_and_bounded() {
    let mut coordinator = Coordinator::new(Vec::<String>::new());
    let artifact = ArtifactEnvelope::new(b"kernel-bytes".to_vec());
    let content_hash = artifact.content_hash.clone();

    coordinator.add_artifact(artifact).unwrap();

    assert_eq!(
        coordinator.fetch_artifact(&content_hash, 64).unwrap(),
        b"kernel-bytes".to_vec()
    );
    assert_eq!(
        coordinator.fetch_artifact(&content_hash, 4),
        Err(CoordinatorError::ArtifactTooLarge)
    );

    let mut tampered = ArtifactEnvelope::new(b"trusted".to_vec());
    tampered.bytes = b"modified".to_vec();
    assert_eq!(
        coordinator.add_artifact(tampered),
        Err(CoordinatorError::ArtifactHashMismatch)
    );
}

#[test]
fn tampered_expired_and_duplicate_results_are_rejected() {
    let mut coordinator = Coordinator::new(["trusted".to_owned()]);
    let job = JobEnvelope::new("job", "hash", ["cpu".to_owned()], "secret", 10);
    let result = WorkerResult::new("job", "result", "secret");

    assert_eq!(
        coordinator.submit_result(&job, &result, "secret", 11),
        Err(CoordinatorError::ExpiredLease)
    );
    assert_eq!(
        coordinator.submit_result(
            &job,
            &WorkerResult::new("job", "result", "bad"),
            "secret",
            1
        ),
        Err(CoordinatorError::Tampered)
    );
    assert!(coordinator
        .submit_result(&job, &result, "secret", 1)
        .is_ok());
    assert_eq!(
        coordinator.submit_result(&job, &result, "secret", 1),
        Err(CoordinatorError::DuplicateResult)
    );
}

#[test]
fn cancellation_and_default_sandbox_policy_are_safe() {
    let mut coordinator = Coordinator::new(Vec::<String>::new());
    coordinator.cancel("job");

    assert!(coordinator.is_cancelled("job"));
    let policy = SandboxPolicy::default();
    assert!(policy.has(SandboxControl::NetworkDisabled));
    assert!(policy.has(SandboxControl::DenyHostEnv));
    assert!(policy.has(SandboxControl::DenyDockerSocket));
    assert!(policy.has(SandboxControl::ReadOnlyArtifacts));
}
