//! Authenticated coordinator contracts.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use atlas_worker::{WorkerCapabilities, WorkerRegistration};
use rusqlite::{params, Connection};

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

/// Content-addressed artifact stored by the coordinator for worker fetches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactEnvelope {
    /// Stable content hash.
    pub content_hash: String,
    /// Artifact bytes.
    pub bytes: Vec<u8>,
}

impl ArtifactEnvelope {
    /// Creates a content-addressed artifact envelope.
    #[must_use]
    pub fn new(bytes: Vec<u8>) -> Self {
        let content_hash = artifact_hash(&bytes);
        Self {
            content_hash,
            bytes,
        }
    }

    /// Verifies that the stored hash matches the artifact bytes.
    #[must_use]
    pub fn verify(&self) -> bool {
        self.content_hash == artifact_hash(&self.bytes)
    }
}

/// Coordinator error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoordinatorError {
    /// Worker certificate is not trusted.
    UntrustedCertificate,
    /// Transport did not present a client certificate.
    MissingClientCertificate,
    /// The transport endpoint identity does not match the configured server.
    ServerIdentityMismatch,
    /// Job signature or result signature is invalid.
    Tampered,
    /// Lease expired.
    ExpiredLease,
    /// Result was already accepted.
    DuplicateResult,
    /// No worker satisfies required capabilities.
    NoCapableWorker,
    /// Worker is not registered.
    UnknownWorker,
    /// Requested artifact was not found.
    ArtifactNotFound,
    /// Requested artifact exceeds the transfer bound.
    ArtifactTooLarge,
    /// Artifact content does not match its content hash.
    ArtifactHashMismatch,
}

/// Coordinator state.
#[derive(Debug, Clone, Default)]
pub struct Coordinator {
    trusted_certificates: BTreeSet<String>,
    workers: BTreeMap<String, WorkerRegistration>,
    accepted_results: BTreeSet<String>,
    cancelled_jobs: BTreeSet<String>,
    active_leases: BTreeMap<String, String>,
    last_heartbeats: BTreeMap<String, u64>,
    artifacts: BTreeMap<String, ArtifactEnvelope>,
}

/// Durable coordinator snapshot used for restart recovery.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoordinatorSnapshot {
    /// Trusted certificate fingerprints.
    pub trusted_certificates: BTreeSet<String>,
    /// Registered workers by id.
    pub workers: BTreeMap<String, WorkerRegistration>,
    /// Accepted job ids.
    pub accepted_results: BTreeSet<String>,
    /// Cancelled job ids.
    pub cancelled_jobs: BTreeSet<String>,
    /// Active job leases keyed by job id with worker id values.
    pub active_leases: BTreeMap<String, String>,
    /// Last heartbeat tick keyed by worker id.
    pub last_heartbeats: BTreeMap<String, u64>,
    /// Content-addressed artifacts keyed by content hash.
    pub artifacts: BTreeMap<String, ArtifactEnvelope>,
}

/// Peer identity observed after a mutually-authenticated transport handshake.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MutuallyAuthenticatedPeer {
    /// Client certificate fingerprint, if a client certificate was presented.
    pub client_certificate_fingerprint: Option<String>,
    /// Server name used by the client.
    pub server_name: String,
    /// Server certificate fingerprint observed by the client.
    pub server_certificate_fingerprint: String,
}

/// Coordinator mutual-authentication policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MutualTlsTransportPolicy {
    trusted_client_certificates: BTreeSet<String>,
    expected_server_name: String,
    expected_server_certificate_fingerprint: String,
}

impl MutualTlsTransportPolicy {
    /// Authorizes a peer identity captured from a transport handshake.
    ///
    /// # Errors
    ///
    /// Returns an error when the client certificate is missing or untrusted, or
    /// when the server identity does not match the configured coordinator.
    pub fn authorize(&self, peer: &MutuallyAuthenticatedPeer) -> Result<(), CoordinatorError> {
        if peer.server_name != self.expected_server_name
            || peer.server_certificate_fingerprint != self.expected_server_certificate_fingerprint
        {
            return Err(CoordinatorError::ServerIdentityMismatch);
        }
        let Some(client_certificate) = &peer.client_certificate_fingerprint else {
            return Err(CoordinatorError::MissingClientCertificate);
        };
        if !self
            .trusted_client_certificates
            .contains(client_certificate)
        {
            return Err(CoordinatorError::UntrustedCertificate);
        }
        Ok(())
    }
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
            active_leases: BTreeMap::new(),
            last_heartbeats: BTreeMap::new(),
            artifacts: BTreeMap::new(),
        }
    }

    /// Restores a coordinator from a durable snapshot.
    #[must_use]
    pub fn restore(snapshot: CoordinatorSnapshot) -> Self {
        Self {
            trusted_certificates: snapshot.trusted_certificates,
            workers: snapshot.workers,
            accepted_results: snapshot.accepted_results,
            cancelled_jobs: snapshot.cancelled_jobs,
            active_leases: snapshot.active_leases,
            last_heartbeats: snapshot.last_heartbeats,
            artifacts: snapshot.artifacts,
        }
    }

    /// Creates a durable snapshot for restart recovery.
    #[must_use]
    pub fn snapshot(&self) -> CoordinatorSnapshot {
        CoordinatorSnapshot {
            trusted_certificates: self.trusted_certificates.clone(),
            workers: self.workers.clone(),
            accepted_results: self.accepted_results.clone(),
            cancelled_jobs: self.cancelled_jobs.clone(),
            active_leases: self.active_leases.clone(),
            last_heartbeats: self.last_heartbeats.clone(),
            artifacts: self.artifacts.clone(),
        }
    }

    /// Creates a mutual-authentication policy for coordinator RPC transport.
    #[must_use]
    pub fn mutual_tls_policy(
        &self,
        expected_server_name: impl Into<String>,
        expected_server_certificate_fingerprint: impl Into<String>,
    ) -> MutualTlsTransportPolicy {
        MutualTlsTransportPolicy {
            trusted_client_certificates: self.trusted_certificates.clone(),
            expected_server_name: expected_server_name.into(),
            expected_server_certificate_fingerprint: expected_server_certificate_fingerprint.into(),
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

    /// Records a worker heartbeat and updates advertised capabilities.
    ///
    /// # Errors
    ///
    /// Returns an error if the worker is not registered.
    pub fn heartbeat(
        &mut self,
        worker_id: &str,
        capabilities: WorkerCapabilities,
        tick: u64,
    ) -> Result<(), CoordinatorError> {
        let Some(worker) = self.workers.get_mut(worker_id) else {
            return Err(CoordinatorError::UnknownWorker);
        };
        worker.capabilities = capabilities;
        self.last_heartbeats.insert(worker_id.to_owned(), tick);
        Ok(())
    }

    /// Returns the last heartbeat tick for a worker.
    #[must_use]
    pub fn last_heartbeat(&self, worker_id: &str) -> Option<u64> {
        self.last_heartbeats.get(worker_id).copied()
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

    /// Leases a job to the least-capable matching worker and records the active
    /// lease for restart/requeue handling.
    ///
    /// # Errors
    ///
    /// Returns an error if no worker can run the job.
    pub fn lease_job(&mut self, job: &JobEnvelope) -> Result<String, CoordinatorError> {
        let worker_id = self.schedule(job)?;
        self.active_leases.insert(job.id.clone(), worker_id.clone());
        Ok(worker_id)
    }

    /// Returns the worker id holding an active job lease.
    #[must_use]
    pub fn active_lease(&self, job_id: &str) -> Option<&str> {
        self.active_leases.get(job_id).map(String::as_str)
    }

    /// Removes a disconnected worker and releases its active job leases for
    /// requeue.
    #[must_use]
    pub fn worker_disconnected(&mut self, worker_id: &str) -> Vec<String> {
        self.workers.remove(worker_id);
        self.last_heartbeats.remove(worker_id);
        let released = self
            .active_leases
            .iter()
            .filter(|(_, leased_worker)| leased_worker.as_str() == worker_id)
            .map(|(job_id, _)| job_id.clone())
            .collect::<Vec<_>>();
        for job_id in &released {
            self.active_leases.remove(job_id);
        }
        released
    }

    /// Adds a content-addressed artifact for bounded worker fetches.
    ///
    /// # Errors
    ///
    /// Returns an error when the content hash does not match the artifact bytes.
    pub fn add_artifact(&mut self, artifact: ArtifactEnvelope) -> Result<(), CoordinatorError> {
        if !artifact.verify() {
            return Err(CoordinatorError::ArtifactHashMismatch);
        }
        self.artifacts
            .insert(artifact.content_hash.clone(), artifact);
        Ok(())
    }

    /// Fetches a content-addressed artifact subject to a transfer byte limit.
    ///
    /// # Errors
    ///
    /// Returns an error if the artifact is absent, too large, or fails hash
    /// validation.
    pub fn fetch_artifact(
        &self,
        content_hash: &str,
        max_bytes: usize,
    ) -> Result<Vec<u8>, CoordinatorError> {
        let artifact = self
            .artifacts
            .get(content_hash)
            .ok_or(CoordinatorError::ArtifactNotFound)?;
        if artifact.bytes.len() > max_bytes {
            return Err(CoordinatorError::ArtifactTooLarge);
        }
        if !artifact.verify() {
            return Err(CoordinatorError::ArtifactHashMismatch);
        }
        Ok(artifact.bytes.clone())
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

/// SQLite-backed durable coordinator lease state.
pub struct SqliteLeaseStore {
    connection: Connection,
}

impl SqliteLeaseStore {
    /// Opens or creates a lease-state database.
    ///
    /// # Errors
    ///
    /// Returns an error when `SQLite` cannot open the database or initialize the
    /// schema.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, String> {
        let connection = Connection::open(path).map_err(|error| error.to_string())?;
        connection
            .execute_batch(
                "
                CREATE TABLE IF NOT EXISTS trusted_certificates (
                    fingerprint TEXT PRIMARY KEY NOT NULL
                );
                CREATE TABLE IF NOT EXISTS workers (
                    worker_id TEXT PRIMARY KEY NOT NULL,
                    certificate_fingerprint TEXT NOT NULL,
                    cpu_cores INTEGER NOT NULL,
                    memory_mib INTEGER NOT NULL,
                    labels TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS accepted_results (
                    job_id TEXT PRIMARY KEY NOT NULL
                );
                CREATE TABLE IF NOT EXISTS cancelled_jobs (
                    job_id TEXT PRIMARY KEY NOT NULL
                );
                CREATE TABLE IF NOT EXISTS active_leases (
                    job_id TEXT PRIMARY KEY NOT NULL,
                    worker_id TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS last_heartbeats (
                    worker_id TEXT PRIMARY KEY NOT NULL,
                    tick INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    content_hash TEXT PRIMARY KEY NOT NULL,
                    bytes BLOB NOT NULL
                );
                ",
            )
            .map_err(|error| error.to_string())?;
        Ok(Self { connection })
    }

    /// Saves a complete coordinator snapshot.
    ///
    /// # Errors
    ///
    /// Returns an error when `SQLite` rejects a statement or value.
    pub fn save_snapshot(&self, snapshot: &CoordinatorSnapshot) -> Result<(), String> {
        self.connection
            .execute_batch(
                "
                DELETE FROM trusted_certificates;
                DELETE FROM workers;
                DELETE FROM accepted_results;
                DELETE FROM cancelled_jobs;
                DELETE FROM active_leases;
                DELETE FROM last_heartbeats;
                DELETE FROM artifacts;
                ",
            )
            .map_err(|error| error.to_string())?;

        for fingerprint in &snapshot.trusted_certificates {
            self.connection
                .execute(
                    "INSERT INTO trusted_certificates (fingerprint) VALUES (?1)",
                    params![fingerprint],
                )
                .map_err(|error| error.to_string())?;
        }
        for worker in snapshot.workers.values() {
            self.connection
                .execute(
                    "INSERT INTO workers (worker_id, certificate_fingerprint, cpu_cores, memory_mib, labels)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![
                        &worker.worker_id,
                        &worker.certificate_fingerprint,
                        worker.capabilities.cpu_cores,
                        worker.capabilities.memory_mib,
                        encode_labels(&worker.capabilities.labels),
                    ],
                )
                .map_err(|error| error.to_string())?;
        }
        insert_string_set(
            &self.connection,
            "accepted_results",
            "job_id",
            &snapshot.accepted_results,
        )?;
        insert_string_set(
            &self.connection,
            "cancelled_jobs",
            "job_id",
            &snapshot.cancelled_jobs,
        )?;
        for (job_id, worker_id) in &snapshot.active_leases {
            self.connection
                .execute(
                    "INSERT INTO active_leases (job_id, worker_id) VALUES (?1, ?2)",
                    params![job_id, worker_id],
                )
                .map_err(|error| error.to_string())?;
        }
        for (worker_id, tick) in &snapshot.last_heartbeats {
            self.connection
                .execute(
                    "INSERT INTO last_heartbeats (worker_id, tick) VALUES (?1, ?2)",
                    params![worker_id, tick],
                )
                .map_err(|error| error.to_string())?;
        }
        for artifact in snapshot.artifacts.values() {
            self.connection
                .execute(
                    "INSERT INTO artifacts (content_hash, bytes) VALUES (?1, ?2)",
                    params![&artifact.content_hash, &artifact.bytes],
                )
                .map_err(|error| error.to_string())?;
        }
        Ok(())
    }

    /// Loads a complete coordinator snapshot.
    ///
    /// # Errors
    ///
    /// Returns an error when `SQLite` rejects a query or row value.
    pub fn load_snapshot(&self) -> Result<CoordinatorSnapshot, String> {
        Ok(CoordinatorSnapshot {
            trusted_certificates: query_string_set(
                &self.connection,
                "SELECT fingerprint FROM trusted_certificates",
            )?,
            workers: query_workers(&self.connection)?,
            accepted_results: query_string_set(
                &self.connection,
                "SELECT job_id FROM accepted_results",
            )?,
            cancelled_jobs: query_string_set(
                &self.connection,
                "SELECT job_id FROM cancelled_jobs",
            )?,
            active_leases: query_string_map(
                &self.connection,
                "SELECT job_id, worker_id FROM active_leases",
            )?,
            last_heartbeats: query_u64_map(
                &self.connection,
                "SELECT worker_id, tick FROM last_heartbeats",
            )?,
            artifacts: query_artifacts(&self.connection)?,
        })
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

fn insert_string_set(
    connection: &Connection,
    table: &str,
    column: &str,
    values: &BTreeSet<String>,
) -> Result<(), String> {
    let sql = format!("INSERT INTO {table} ({column}) VALUES (?1)");
    for value in values {
        connection
            .execute(&sql, params![value])
            .map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn query_string_set(connection: &Connection, sql: &str) -> Result<BTreeSet<String>, String> {
    let mut statement = connection.prepare(sql).map_err(|error| error.to_string())?;
    let rows = statement
        .query_map([], |row| row.get::<_, String>(0))
        .map_err(|error| error.to_string())?;
    let mut values = BTreeSet::new();
    for row in rows {
        values.insert(row.map_err(|error| error.to_string())?);
    }
    Ok(values)
}

fn query_string_map(
    connection: &Connection,
    sql: &str,
) -> Result<BTreeMap<String, String>, String> {
    let mut statement = connection.prepare(sql).map_err(|error| error.to_string())?;
    let rows = statement
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .map_err(|error| error.to_string())?;
    let mut values = BTreeMap::new();
    for row in rows {
        let (key, value) = row.map_err(|error| error.to_string())?;
        values.insert(key, value);
    }
    Ok(values)
}

fn query_u64_map(connection: &Connection, sql: &str) -> Result<BTreeMap<String, u64>, String> {
    let mut statement = connection.prepare(sql).map_err(|error| error.to_string())?;
    let rows = statement
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, u64>(1)?))
        })
        .map_err(|error| error.to_string())?;
    let mut values = BTreeMap::new();
    for row in rows {
        let (key, value) = row.map_err(|error| error.to_string())?;
        values.insert(key, value);
    }
    Ok(values)
}

fn query_workers(connection: &Connection) -> Result<BTreeMap<String, WorkerRegistration>, String> {
    let mut statement = connection
        .prepare(
            "SELECT worker_id, certificate_fingerprint, cpu_cores, memory_mib, labels FROM workers",
        )
        .map_err(|error| error.to_string())?;
    let rows = statement
        .query_map([], |row| {
            let worker_id = row.get::<_, String>(0)?;
            Ok(WorkerRegistration {
                worker_id,
                certificate_fingerprint: row.get(1)?,
                capabilities: WorkerCapabilities {
                    cpu_cores: row.get(2)?,
                    memory_mib: row.get(3)?,
                    labels: decode_labels(&row.get::<_, String>(4)?),
                },
            })
        })
        .map_err(|error| error.to_string())?;
    let mut workers = BTreeMap::new();
    for row in rows {
        let worker = row.map_err(|error| error.to_string())?;
        workers.insert(worker.worker_id.clone(), worker);
    }
    Ok(workers)
}

fn query_artifacts(connection: &Connection) -> Result<BTreeMap<String, ArtifactEnvelope>, String> {
    let mut statement = connection
        .prepare("SELECT content_hash, bytes FROM artifacts")
        .map_err(|error| error.to_string())?;
    let rows = statement
        .query_map([], |row| {
            let content_hash = row.get::<_, String>(0)?;
            Ok(ArtifactEnvelope {
                content_hash,
                bytes: row.get(1)?,
            })
        })
        .map_err(|error| error.to_string())?;
    let mut artifacts = BTreeMap::new();
    for row in rows {
        let artifact = row.map_err(|error| error.to_string())?;
        artifacts.insert(artifact.content_hash.clone(), artifact);
    }
    Ok(artifacts)
}

fn encode_labels(labels: &BTreeSet<String>) -> String {
    labels.iter().cloned().collect::<Vec<_>>().join("\u{1f}")
}

fn decode_labels(value: &str) -> BTreeSet<String> {
    value
        .split('\u{1f}')
        .filter(|label| !label.is_empty())
        .map(str::to_owned)
        .collect()
}

fn artifact_hash(bytes: &[u8]) -> String {
    let mut state = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        state ^= u64::from(*byte);
        state = state.wrapping_mul(0x0100_0000_01b3);
    }
    format!("{state:016x}")
}

/// Convenience constructor for worker capabilities.
#[must_use]
pub fn caps(labels: &[&str]) -> WorkerCapabilities {
    WorkerCapabilities::new(labels.iter().map(|label| (*label).to_owned()))
}
