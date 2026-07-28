//! Cooperative scheduler and backend protocol boundary.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use atlas_planner::Portfolio;

/// Backend execution event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    /// Strategy started.
    Started(String),
    /// Strategy completed.
    Completed(String),
    /// Strategy crashed and was isolated.
    BackendCrashed {
        /// Strategy whose backend failed.
        strategy: String,
        /// Failure message captured at the adapter boundary.
        message: String,
    },
    /// Strategy was skipped or stopped due to cancellation.
    Cancelled(String),
}

/// Backend adapter contract.
pub trait Backend {
    /// Runs a named strategy.
    ///
    /// # Errors
    ///
    /// Returns an error when the backend crashes or rejects the job.
    fn run(&mut self, strategy: &str, cancellation: &CancellationToken) -> Result<(), String>;
}

/// Cooperative cancellation token.
#[derive(Debug, Clone, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    /// Creates a token.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Requests cancellation.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    /// Returns whether cancellation was requested.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }
}

/// Sequential scheduler preserving crash isolation and cancellation events.
pub struct Scheduler;

impl Scheduler {
    /// Runs a portfolio against a backend.
    #[must_use]
    pub fn run(
        portfolio: &Portfolio,
        backend: &mut dyn Backend,
        cancellation: &CancellationToken,
    ) -> Vec<Event> {
        let mut events = Vec::new();
        for stage in &portfolio.stages {
            if cancellation.is_cancelled() {
                events.push(Event::Cancelled(stage.name.clone()));
                continue;
            }
            events.push(Event::Started(stage.name.clone()));
            match backend.run(&stage.name, cancellation) {
                Ok(()) => events.push(Event::Completed(stage.name.clone())),
                Err(message) => events.push(Event::BackendCrashed {
                    strategy: stage.name.clone(),
                    message,
                }),
            }
        }
        events
    }
}
