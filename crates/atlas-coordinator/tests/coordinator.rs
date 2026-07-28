//! Coordinator/worker integration tests.

use atlas_coordinator::{caps, Coordinator, CoordinatorError, JobEnvelope, WorkerResult};
use atlas_worker::{SandboxControl, SandboxPolicy, WorkerRegistration};

fn registration(id: &str, cert: &str, labels: &[&str]) -> WorkerRegistration {
    WorkerRegistration {
        worker_id: id.to_owned(),
        certificate_fingerprint: cert.to_owned(),
        capabilities: caps(labels),
    }
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
