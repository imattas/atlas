//! Scheduler contract tests.

use atlas_planner::{Portfolio, Strategy};
use atlas_scheduler::{Backend, CancellationToken, Event, Scheduler};

struct CrashBackend;

impl Backend for CrashBackend {
    fn run(&mut self, strategy: &str, _cancellation: &CancellationToken) -> Result<(), String> {
        if strategy == "crash" {
            Err("backend exited".to_owned())
        } else {
            Ok(())
        }
    }
}

#[test]
fn backend_crashes_are_reported_without_stopping_later_stages() {
    let portfolio = Portfolio {
        stages: vec![
            Strategy {
                name: "crash".to_owned(),
                time_budget_ms: 1,
            },
            Strategy {
                name: "fallback".to_owned(),
                time_budget_ms: 1,
            },
        ],
    };
    let mut backend = CrashBackend;
    let token = CancellationToken::new();

    let events = Scheduler::run(&portfolio, &mut backend, &token);

    assert!(matches!(events[1], Event::BackendCrashed { .. }));
    assert_eq!(events[2], Event::Started("fallback".to_owned()));
    assert_eq!(events[3], Event::Completed("fallback".to_owned()));
}

#[test]
fn cancellation_skips_pending_stages() {
    let portfolio = Portfolio {
        stages: vec![Strategy {
            name: "pending".to_owned(),
            time_budget_ms: 1,
        }],
    };
    let mut backend = CrashBackend;
    let token = CancellationToken::new();
    token.cancel();

    let events = Scheduler::run(&portfolio, &mut backend, &token);

    assert_eq!(events, vec![Event::Cancelled("pending".to_owned())]);
}
