# oxide-conservation

**Conservation law verification for GPU computations.**

A Rust library that checks whether energy, mass, and information quantities are conserved across GPU kernel boundaries — catching numerical drift, rounding errors, and algorithmic bugs before they cascade through your compute pipeline.

## Why Conservation Verification?

GPU kernels manipulate massive arrays of floating-point values in parallel. Between kernel launches, data flows through global memory, shared memory, registers, and back again. At every boundary, subtle bugs can introduce drift:

- **Floating-point rounding** in reductions, scans, and matrix operations
- **Race conditions** in atomic operations that lose updates
- **Algorithmic errors** where a kernel inadvertently drops or duplicates data
- **Numerical instability** where error compounds across thousands of iterations

In scientific computing, physics simulation, and financial modeling, these bugs are catastrophic. A mass conservation violation of 0.01% in a fluid dynamics simulation might be invisible in a single frame but render a 10,000-step integration completely wrong.

**Conservation verification is the assertion layer for numerical correctness.** Just as you assert invariants in software, you should assert conservation invariants in compute pipelines.

## The Math

### Ternary Verification

For any conserved quantity Q measured before and after a kernel execution:

```
Δ = |Q_after - Q_before|
```

The ternary verdict is:

| Condition           | Verdict      | Value |
|---------------------|--------------|-------|
| Δ = 0               | Conserved    | +1    |
| 0 < Δ ≤ ε           | Approximate  | 0     |
| Δ > ε               | Violated     | −1    |

Where ε (epsilon) is a configurable tolerance that reflects the expected numerical precision of your computation.

### Cumulative Drift

Across N kernel executions, cumulative drift is:

```
D_total = Σᵢ |Δᵢ|
```

When `D_total > threshold`, an alert is raised. This catches slow leaks that would pass any single-kernel check but accumulate over a full simulation run.

### Conservation Laws

Three built-in laws are provided:

- **Energy** — Total energy before = total energy after. Critical for physics simulations, Hamiltonian systems, and N-body integrators.
- **Mass** — Total mass/particle count is preserved. Essential for fluid dynamics, particle transport, and chemical reaction modeling.
- **Information** — Entropy bounds, data integrity, bit-level preservation. Relevant for compression, encryption, and lossless transform pipelines.

Custom laws can be defined for domain-specific invariants (angular momentum, charge, baryon number, etc.).

## Key Types

### `ConservationLaw`

```rust
pub enum ConservationLaw {
    Energy,
    Mass,
    Information,
    Custom(String),
}
```

The law being verified. `Custom` variants let you define domain-specific invariants like `"angular_momentum"` or `"electric_charge"`.

### `Ternary`

```rust
pub enum Ternary {
    Conserved = 1,   // exact match
    Approximate = 0,  // within epsilon
    Violated = -1,    // exceeds threshold
}
```

Three-valued logic that distinguishes "perfectly conserved" from "close enough" from "broken." This is more informative than a simple boolean pass/fail.

### `ConservationMonitor`

Takes snapshots of quantities before kernel execution and verifies them after:

```rust
let mut monitor = ConservationMonitor::new(1e-6); // epsilon
monitor.snapshot(ConservationLaw::Energy, "nbody_step", total_energy);
// ... launch GPU kernel ...
let result = monitor.verify(&ConservationLaw::Energy, "nbody_step", new_total_energy);
assert!(result.unwrap().verdict.is_ok());
```

### `VerificationResult`

Holds the full picture: law, before/after values, delta, epsilon, and the ternary verdict.

### `ConservationBudget`

Tracks cumulative drift across an entire compute pipeline:

```rust
let mut budget = ConservationBudget::new(0.1); // alert threshold

for step in 0..10000 {
    let result = monitor.verify(&ConservationLaw::Energy, "sim", energy_after);
    budget.record(&result.unwrap());
    
    if let Some(alert) = budget.check_alert() {
        eprintln!("WARNING: {}", alert);
        break;
    }
}

println!("Average drift: {}", budget.average_drift());
println!("Violation rate: {:.2}%", budget.violation_rate() * 100.0);
```

### `Alert`

Raised when cumulative drift exceeds the budget threshold. Includes the total drift, the threshold, and the worst-offending conservation law.

## Usage

Add to `Cargo.toml`:

```toml
[dependencies]
oxide-conservation = "0.1"
```

### Basic Verification

```rust
use oxide_conservation::{ConservationLaw, VerificationResult, Ternary};

let result = VerificationResult::check(
    ConservationLaw::Mass,
    1000.0,   // before
    999.9998, // after
    0.001,    // epsilon
);

assert_eq!(result.verdict, Ternary::Approximate);
println!("{}", result); // [ 0 (approximate)] 1000 → 999.9998 (Δ=0.0002, ε=0.001)
```

### Multi-Kernel Pipeline

```rust
use oxide_conservation::{ConservationMonitor, ConservationLaw, ConservationBudget};

let mut monitor = ConservationMonitor::new(1e-9);
let mut budget = ConservationBudget::new(1e-3);

// Snapshot before kernel chain
monitor.snapshot(ConservationLaw::Energy, "fft_forward", energy_in);
monitor.snapshot(ConservationLaw::Information, "fft_forward", entropy_in);

// ... run FFT kernel on GPU ...

// Verify after
let energy_result = monitor.verify(&ConservationLaw::Energy, "fft_forward", energy_out);
let info_result = monitor.verify(&ConservationLaw::Information, "fft_forward", entropy_out);

budget.record(&energy_result.unwrap());
budget.record(&info_result.unwrap());

if let Some(alert) = budget.check_alert() {
    log::error!("Conservation drift alert: {}", alert);
}
```

### Batch Verification

```rust
monitor.snapshot(ConservationLaw::Energy, "kernel_a", 42.0);
monitor.snapshot(ConservationLaw::Mass, "kernel_b", 100.0);
monitor.snapshot(ConservationLaw::Custom("charge".into()), "kernel_c", 0.0);

// Provide after-values for all pending snapshots
let results = monitor.verify_all(|law, label| {
    match label {
        "kernel_a" => 42.0000001,
        "kernel_b" => 100.0,
        "kernel_c" => 0.0,
        _ => 0.0,
    }
});

for r in &results {
    println!("{}: {}", r.law, r.verdict);
}
```

## Statistics

`ConservationBudget` provides aggregate statistics across your entire workload:

| Method                | Description                                      |
|-----------------------|--------------------------------------------------|
| `verification_count()` | Total number of verifications recorded           |
| `violation_count()`   | Number of verifications that returned `Violated` |
| `violation_rate()`    | Fraction of verifications that were violations   |
| `average_drift()`     | Mean absolute drift across all verifications     |
| `drift_for(law)`      | Cumulative drift for a specific conservation law |
| `total_drift()`       | Sum of drift across all laws                     |

## Connection to the Oxide Stack

`oxide-conservation` is part of the **Oxide** family of crates for robust GPU compute in Rust:

- **oxide-conservation** — Conservation law verification (this crate)
- Works alongside any GPU compute framework (wgpu, CUDA, OpenCL)
- Designed to integrate into CI pipelines for numerical regression testing
- Zero dependencies — add verification without bloating your build

The Oxide philosophy: **numerical correctness is not optional.** Every GPU compute pipeline should verify its invariants. This crate makes it easy.

## License

MIT
