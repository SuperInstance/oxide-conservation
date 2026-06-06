//! # oxide-conservation
//!
//! Conservation law verification for GPU computations.
//!
//! Checks that energy, mass, and information are conserved across kernel
//! boundaries using ternary verification results:
//! - `+1` → conserved (exact match)
//! - ` 0` → approximate (within epsilon)
//! - `-1` → violated (exceeds threshold)

use std::fmt;

/// Ternary verification result for conservation checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Ternary {
    /// Quantity is exactly conserved.
    Conserved = 1,
    /// Quantity is approximately conserved (within epsilon).
    Approximate = 0,
    /// Conservation is violated beyond threshold.
    Violated = -1,
}

impl Ternary {
    /// Create from an `i8` value.
    pub fn from_i8(v: i8) -> Self {
        match v {
            1 => Ternary::Conserved,
            0 => Ternary::Approximate,
            _ => Ternary::Violated,
        }
    }

    /// Returns `true` if conservation holds (exact or approximate).
    pub fn is_ok(self) -> bool {
        self != Ternary::Violated
    }
}

impl fmt::Display for Ternary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Ternary::Conserved => write!(f, "+1 (conserved)"),
            Ternary::Approximate => write!(f, " 0 (approximate)"),
            Ternary::Violated => write!(f, "-1 (violated)"),
        }
    }
}

/// Kinds of conservation laws that can be verified.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ConservationLaw {
    /// Conservation of energy: Σ E_in == Σ E_out
    Energy,
    /// Conservation of mass: Σ m_in == Σ m_out
    Mass,
    /// Conservation of information (e.g., entropy bounds, data integrity)
    Information,
    /// A user-defined conservation law with a label.
    Custom(String),
}

impl fmt::Display for ConservationLaw {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConservationLaw::Energy => write!(f, "energy"),
            ConservationLaw::Mass => write!(f, "mass"),
            ConservationLaw::Information => write!(f, "information"),
            ConservationLaw::Custom(name) => write!(f, "custom({})", name),
        }
    }
}

/// Result of a single conservation verification.
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// The law that was checked.
    pub law: ConservationLaw,
    /// Quantity before kernel execution.
    pub before: f64,
    /// Quantity after kernel execution.
    pub after: f64,
    /// Absolute difference `|after - before|`.
    pub delta: f64,
    /// Epsilon threshold for approximate conservation.
    pub epsilon: f64,
    /// Ternary verdict.
    pub verdict: Ternary,
}

impl VerificationResult {
    /// Build a verification result by comparing before/after values.
    pub fn check(law: ConservationLaw, before: f64, after: f64, epsilon: f64) -> Self {
        let delta = (after - before).abs();
        let verdict = if delta == 0.0 {
            Ternary::Conserved
        } else if delta <= epsilon {
            Ternary::Approximate
        } else {
            Ternary::Violated
        };
        VerificationResult {
            law,
            before,
            after,
            delta,
            epsilon,
            verdict,
        }
    }
}

impl fmt::Display for VerificationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {} → {} (Δ={}, ε={})",
            self.verdict, self.before, self.after, self.delta, self.epsilon
        )
    }
}

/// Monitors conservation of tracked quantities across kernel boundaries.
#[derive(Debug, Clone)]
pub struct ConservationMonitor {
    /// Epsilon for approximate conservation checks.
    epsilon: f64,
    /// Named quantities: (law, label) → value before kernel.
    snapshots: Vec<(ConservationLaw, String, f64)>,
    /// Completed verification results.
    results: Vec<VerificationResult>,
}

impl ConservationMonitor {
    /// Create a new monitor with the given epsilon threshold.
    pub fn new(epsilon: f64) -> Self {
        ConservationMonitor {
            epsilon,
            snapshots: Vec::new(),
            results: Vec::new(),
        }
    }

    /// Record a quantity *before* kernel execution.
    pub fn snapshot(&mut self, law: ConservationLaw, label: impl Into<String>, value: f64) {
        self.snapshots.push((law, label.into(), value));
    }

    /// Verify a previously-snapshotted quantity *after* kernel execution.
    ///
    /// Returns `None` if no matching snapshot exists.
    pub fn verify(&mut self, law: &ConservationLaw, label: &str, after_value: f64) -> Option<VerificationResult> {
        let idx = self.snapshots.iter().position(|(l, lb, _)| l == law && lb == label)?;
        let (law, _, before) = self.snapshots.remove(idx);
        let result = VerificationResult::check(law, before, after_value, self.epsilon);
        self.results.push(result.clone());
        Some(result)
    }

    /// Verify all remaining snapshots against a closure that provides the "after" value.
    pub fn verify_all<F>(&mut self, mut after_fn: F) -> Vec<VerificationResult>
    where
        F: FnMut(&ConservationLaw, &str) -> f64,
    {
        let snapshots = std::mem::take(&mut self.snapshots);
        let mut batch = Vec::new();
        for (law, label, before) in snapshots {
            let after = after_fn(&law, &label);
            let result = VerificationResult::check(law, before, after, self.epsilon);
            batch.push(result.clone());
            self.results.push(result);
        }
        batch
    }

    /// Return all verification results collected so far.
    pub fn results(&self) -> &[VerificationResult] {
        &self.results
    }

    /// Number of unverified snapshots remaining.
    pub fn pending(&self) -> usize {
        self.snapshots.len()
    }

    /// Reset the monitor for a new round of checks.
    pub fn reset(&mut self) {
        self.snapshots.clear();
        self.results.clear();
    }
}

/// Tracks cumulative conservation drift across multiple kernel executions.
#[derive(Debug, Clone)]
pub struct ConservationBudget {
    /// Per-law cumulative drift.
    drifts: Vec<(ConservationLaw, f64)>,
    /// Threshold beyond which an alert is raised.
    alert_threshold: f64,
    /// Statistics.
    total_verifications: u64,
    total_violations: u64,
    drift_history: Vec<f64>,
}

impl ConservationBudget {
    /// Create a new budget with the given alert threshold.
    pub fn new(alert_threshold: f64) -> Self {
        ConservationBudget {
            drifts: Vec::new(),
            alert_threshold,
            total_verifications: 0,
            total_violations: 0,
            drift_history: Vec::new(),
        }
    }

    /// Record a verification result, accumulating drift.
    pub fn record(&mut self, result: &VerificationResult) {
        self.total_verifications += 1;
        if result.verdict == Ternary::Violated {
            self.total_violations += 1;
        }
        self.drift_history.push(result.delta);
        if let Some(entry) = self.drifts.iter_mut().find(|(l, _)| l == &result.law) {
            entry.1 += result.delta;
        } else {
            self.drifts.push((result.law.clone(), result.delta));
        }
    }

    /// Record multiple results at once.
    pub fn record_all(&mut self, results: &[VerificationResult]) {
        for r in results {
            self.record(r);
        }
    }

    /// Get the cumulative drift for a specific law.
    pub fn drift_for(&self, law: &ConservationLaw) -> f64 {
        self.drifts
            .iter()
            .find(|(l, _)| l == law)
            .map(|&(_, d)| d)
            .unwrap_or(0.0)
    }

    /// Get the total cumulative drift across all laws.
    pub fn total_drift(&self) -> f64 {
        self.drifts.iter().map(|(_, d)| d).sum()
    }

    /// Check whether cumulative drift exceeds the alert threshold.
    pub fn check_alert(&self) -> Option<Alert> {
        let total = self.total_drift();
        if total > self.alert_threshold {
            Some(Alert {
                total_drift: total,
                threshold: self.alert_threshold,
                worst_law: self.drifts.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).map(|(l, _)| l.clone()),
            })
        } else {
            None
        }
    }

    /// Total number of verifications recorded.
    pub fn verification_count(&self) -> u64 {
        self.total_verifications
    }

    /// Total number of violations recorded.
    pub fn violation_count(&self) -> u64 {
        self.total_violations
    }

    /// Average drift per verification.
    pub fn average_drift(&self) -> f64 {
        if self.drift_history.is_empty() {
            0.0
        } else {
            self.drift_history.iter().sum::<f64>() / self.drift_history.len() as f64
        }
    }

    /// Violation rate as a fraction (0.0 to 1.0).
    pub fn violation_rate(&self) -> f64 {
        if self.total_verifications == 0 {
            0.0
        } else {
            self.total_violations as f64 / self.total_verifications as f64
        }
    }
}

/// An alert raised when cumulative drift exceeds the budget threshold.
#[derive(Debug, Clone)]
pub struct Alert {
    /// Total cumulative drift.
    pub total_drift: f64,
    /// Configured alert threshold.
    pub threshold: f64,
    /// The conservation law with the worst drift.
    pub worst_law: Option<ConservationLaw>,
}

impl fmt::Display for Alert {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CONSERVATION ALERT: drift {:.6} exceeds threshold {:.6}",
            self.total_drift, self.threshold
        )?;
        if let Some(ref law) = self.worst_law {
            write!(f, " (worst: {})", law)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ternary_exact_conservation() {
        let result = VerificationResult::check(
            ConservationLaw::Energy,
            100.0,
            100.0,
            1e-9,
        );
        assert_eq!(result.verdict, Ternary::Conserved);
        assert_eq!(result.delta, 0.0);
    }

    #[test]
    fn ternary_approximate_conservation() {
        let result = VerificationResult::check(
            ConservationLaw::Mass,
            50.0,
            50.0005,
            0.001,
        );
        assert_eq!(result.verdict, Ternary::Approximate);
        assert!(result.delta > 0.0);
        assert!(result.delta <= 0.001);
    }

    #[test]
    fn ternary_violation() {
        let result = VerificationResult::check(
            ConservationLaw::Information,
            200.0,
            195.0,
            1.0,
        );
        assert_eq!(result.verdict, Ternary::Violated);
        assert_eq!(result.delta, 5.0);
    }

    #[test]
    fn monitor_snapshot_and_verify() {
        let mut mon = ConservationMonitor::new(0.01);
        mon.snapshot(ConservationLaw::Energy, "kernel_a", 42.0);
        assert_eq!(mon.pending(), 1);

        let result = mon.verify(&ConservationLaw::Energy, "kernel_a", 42.005).unwrap();
        assert_eq!(result.verdict, Ternary::Approximate);
        assert_eq!(mon.pending(), 0);
        assert_eq!(mon.results().len(), 1);
    }

    #[test]
    fn monitor_verify_missing_returns_none() {
        let mut mon = ConservationMonitor::new(0.01);
        assert!(mon.verify(&ConservationLaw::Energy, "nope", 0.0).is_none());
    }

    #[test]
    fn monitor_verify_all_batch() {
        let mut mon = ConservationMonitor::new(0.1);
        mon.snapshot(ConservationLaw::Energy, "k1", 10.0);
        mon.snapshot(ConservationLaw::Mass, "k2", 20.0);
        mon.snapshot(ConservationLaw::Information, "k3", 30.0);

        let results = mon.verify_all(|law, _label| match law {
            ConservationLaw::Energy => 10.0,
            ConservationLaw::Mass => 20.05,
            ConservationLaw::Information => 30.2,
            _ => 0.0,
        });

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].verdict, Ternary::Conserved);
        assert_eq!(results[1].verdict, Ternary::Approximate);
        assert_eq!(results[2].verdict, Ternary::Violated);
    }

    #[test]
    fn budget_tracks_drift_and_alerts() {
        let mut budget = ConservationBudget::new(5.0);

        // Three small drifts that accumulate
        for _ in 0..3 {
            let r = VerificationResult::check(ConservationLaw::Energy, 100.0, 98.0, 0.5);
            budget.record(&r);
        }

        assert_eq!(budget.verification_count(), 3);
        assert_eq!(budget.violation_count(), 3); // delta=2.0 > epsilon=0.5
        assert!((budget.drift_for(&ConservationLaw::Energy) - 6.0).abs() < 1e-9);
        assert!(budget.check_alert().is_some()); // 6.0 > 5.0 threshold
    }

    #[test]
    fn budget_statistics() {
        let mut budget = ConservationBudget::new(100.0);

        let r1 = VerificationResult::check(ConservationLaw::Mass, 10.0, 10.0, 0.01);
        let r2 = VerificationResult::check(ConservationLaw::Mass, 10.0, 10.02, 0.01);
        let r3 = VerificationResult::check(ConservationLaw::Mass, 10.0, 10.5, 0.01);
        budget.record_all(&[r1, r2, r3]);

        assert_eq!(budget.verification_count(), 3);
        assert_eq!(budget.violation_count(), 2); // r2 approximate, r3 violated
        let avg = budget.average_drift();
        assert!(avg > 0.0);
        let rate = budget.violation_rate();
        assert!((rate - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn custom_conservation_law() {
        let custom = ConservationLaw::Custom("angular_momentum".into());
        let result = VerificationResult::check(custom.clone(), 7.0, 7.0, 0.0);
        assert_eq!(result.verdict, Ternary::Conserved);
        assert_eq!(result.law, custom);
    }
}
