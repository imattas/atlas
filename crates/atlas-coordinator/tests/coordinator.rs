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
