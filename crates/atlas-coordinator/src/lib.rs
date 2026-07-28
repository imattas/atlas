//! Authenticated coordinator contracts.

use std::collections::{BTreeMap, BTreeSet};

use atlas_worker::{WorkerCapabilities, WorkerRegistration};

/// Signed content-addressed job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JobEnvelope {
    /// Job id.
    pub id: String,
    /// Content hash.
    pub content_hash: String,
    /// Required capabilities.
    pub required: BTreeSet<String>,
    /// Signature over id/hash/scope.
    pub signature: String,
    /// Lease expiry tick.
    pub lease_expires_at: u64,
}

impl JobEnvelope {
    /// Creates a signed job.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        content_hash: impl Into<String>,
        required: impl IntoIterator<Item = String>,
        secret: &str,
        lease_expires_at: u64,
    ) -> Self {
        let id = id.into();
        let content_hash = content_hash.into();
        let required = required.into_iter().collect();
        let signature = sign_parts(&[&id, &content_hash], secret);
        Self {
            id,
            content_hash,
            required,
            signature,
            lease_expires_at,
        }
    }

    /// Verifies the job signature.
    #[must_use]
    pub fn verify(&self, secret: &str) -> bool {
        self.signature == sign_parts(&[&self.id, &self.content_hash], secret)
    }
}

/// Worker result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerResult {
    /// Job id.
    pub job_id: String,
    /// Result hash.
    pub result_hash: String,
    /// Signature over job/result.
    pub signature: String,
}

impl WorkerResult {
    /// Creates a signed result.
    #[must_use]
    pub fn new(job_id: impl Into<String>, result_hash: impl Into<String>, secret: &str) -> Self {
        let job_id = job_id.into();
        let result_hash = result_hash.into();
        let signature = sign_parts(&[&job_id, &result_hash], secret);
        Self {
            job_id,
            result_hash,
            signature,
        }
    }

    /// Verifies the result signature.
    #[must_use]
    pub fn verify(&self, secret: &str) -> bool {
        self.signature == sign_parts(&[&self.job_id, &self.result_hash], secret)
    }
}

/// Coordinator error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoordinatorError {
    /// Worker certificate is not trusted.
    UntrustedCertificate,
    /// Job signature or result signature is invalid.
    Tampered,
    /// Lease expired.
    ExpiredLease,
    /// Result was already accepted.
    DuplicateResult,
    /// No worker satisfies required capabilities.
    NoCapableWorker,
}

/// Coordinator state.
#[derive(Debug, Clone, Default)]
pub struct Coordinator {
    trusted_certificates: BTreeSet<String>,
    workers: BTreeMap<String, WorkerRegistration>,
    accepted_results: BTreeSet<String>,
    cancelled_jobs: BTreeSet<String>,
}

impl Coordinator {
    /// Creates a coordinator with trusted certificate fingerprints.
    #[must_use]
    pub fn new(trusted_certificates: impl IntoIterator<Item = String>) -> Self {
        Self {
            trusted_certificates: trusted_certificates.into_iter().collect(),
            workers: BTreeMap::new(),
            accepted_results: BTreeSet::new(),
            cancelled_jobs: BTreeSet::new(),
        }
    }

    /// Registers a worker after certificate validation.
    ///
    /// # Errors
    ///
    /// Returns an error if the certificate is not trusted.
    pub fn register(&mut self, registration: WorkerRegistration) -> Result<(), CoordinatorError> {
        if !self
            .trusted_certificates
            .contains(&registration.certificate_fingerprint)
        {
            return Err(CoordinatorError::UntrustedCertificate);
        }
        self.workers
            .insert(registration.worker_id.clone(), registration);
        Ok(())
    }

    /// Picks the least-capability worker satisfying a job.
    ///
    /// # Errors
    ///
    /// Returns an error if no worker can run the job.
    pub fn schedule(&self, job: &JobEnvelope) -> Result<String, CoordinatorError> {
        self.workers
            .values()
            .filter(|worker| worker.capabilities.satisfies(&job.required))
            .min_by_key(|worker| worker.capabilities.labels.len())
            .map(|worker| worker.worker_id.clone())
            .ok_or(CoordinatorError::NoCapableWorker)
    }

    /// Accepts a signed result if lease and integrity checks pass.
    ///
    /// # Errors
    ///
    /// Returns an error for tampering, duplicates, or expired leases.
    pub fn submit_result(
        &mut self,
        job: &JobEnvelope,
        result: &WorkerResult,
        secret: &str,
        now: u64,
    ) -> Result<(), CoordinatorError> {
        if now > job.lease_expires_at {
            return Err(CoordinatorError::ExpiredLease);
        }
        if !job.verify(secret) || !result.verify(secret) || result.job_id != job.id {
            return Err(CoordinatorError::Tampered);
        }
        if !self.accepted_results.insert(result.job_id.clone()) {
            return Err(CoordinatorError::DuplicateResult);
        }
        Ok(())
    }

    /// Cancels a job.
    pub fn cancel(&mut self, job_id: impl Into<String>) {
        self.cancelled_jobs.insert(job_id.into());
    }

    /// Returns whether a job is cancelled.
    #[must_use]
    pub fn is_cancelled(&self, job_id: &str) -> bool {
        self.cancelled_jobs.contains(job_id)
    }
}

/// Creates a deterministic test signature.
#[must_use]
pub fn sign_parts(parts: &[&str], secret: &str) -> String {
    let mut state = 0_u64;
    for part in parts.iter().copied().chain([secret]) {
        for byte in part.bytes() {
            state = state.wrapping_mul(131).wrapping_add(u64::from(byte));
        }
    }
    format!("{state:016x}")
}

/// Convenience constructor for worker capabilities.
#[must_use]
pub fn caps(labels: &[&str]) -> WorkerCapabilities {
    WorkerCapabilities::new(labels.iter().map(|label| (*label).to_owned()))
}
