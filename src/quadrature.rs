use crate::definitions::{Function1D, Interval};
use crate::util::make_supporting_points;
use crate::{abs, ln};
use rayon::prelude::*;

/// Results of a quadrature test run.
/// Contains all data for further analysis.
#[derive(Debug, Copy, Clone)]
pub struct QuadratureTestResult {
    /// Resulting integral value
    pub value: f64,
    /// |value - exact|
    pub abs_error: f64,
    /// How many sub intervals were used to obtain the result.
    pub splits_n: usize,
    /// The h corresponding to splits_n
    pub h: f64,
}

/// Type alias for a quadrature method.
pub type QuadratureFormula = fn(f: Function1D, interval: &Interval) -> f64;

/// Implementation of the trapezoid formula
pub fn trapezoid_formula(f: Function1D, interval: &Interval) -> f64 {
    (interval.span() / 2.0) * (f(interval.start()) + f(interval.end()))
}

/// Implementation of the trapezoid formula
pub fn kepler_formula(f: Function1D, interval: &Interval) -> f64 {
    let mid = interval.start() + (1.0 / 2.0) * (interval.end() - interval.start());
    (interval.span() / 6.0) * (f(interval.start()) + 4.0 * f(mid) + f(interval.end()))
}

/// Implementation of the trapezoid formula
pub fn newton_three_eight_formula(f: Function1D, interval: &Interval) -> f64 {
    let mid1 = interval.start() + (1.0 / 3.0) * (interval.end() - interval.start());
    let mid2 = interval.start() + (2.0 / 3.0) * (interval.end() - interval.start());
    (interval.span() / 8.0)
        * (f(interval.start()) + 3.0 * f(mid1) + 3.0 * f(mid2) + f(interval.end()))
}

/// Generic implementation of approximating a function area over an interval.
///
///# Arguments
///
/// * `method` - The method used, e.g. trapezoid_formula, kepler_formula, and so on.
/// * `f` - function to integrate.
/// * `interval` - Interval over which to integrate.
/// * `n_splits` - Number of times the interval is split. This is specified instead of an h or step_size. The result would be h = interval.span() / n_steps.
///
/// # Example
/// ```
/// use ngdl_rust::definitions::{Function1D, Interval};
/// use ngdl_rust::quadrature::{quadrature, newton_three_eight_formula};
///
/// let interval = Interval::new(1.0, 10.0);
/// let f: Function1D = |x| x.ln();
///
/// // Note that we can define a function alias to make using the method more ergonomic.
/// let integrate = |f, interval, n| quadrature(newton_three_eight_formula, f, interval, n);
///
/// dbg!(integrate(f, interval, 1000));
/// ```
pub fn quadrature(
    method: QuadratureFormula,
    f: Function1D,
    interval: Interval,
    n_splits: usize,
) -> f64 {
    let pts = make_supporting_points(n_splits, interval);
    quadrature_with_supporting_points(method, f, &pts)
}

/// Runs the supplied quadrature method for every number of splits from 1 to `up_to_splits`.
pub fn quadrature_test_run(
    method: QuadratureFormula,
    f: Function1D,
    exact: f64,
    interval: Interval,
    up_to_splits: usize,
) -> Vec<QuadratureTestResult> {
    (1..=up_to_splits)
        .into_par_iter()
        .map(|n| {
            let value = quadrature(method, f, interval, n);
            QuadratureTestResult {
                value,
                abs_error: abs!(exact - value),
                splits_n: n,
                h: interval.span() / n as f64,
            }
        })
        .collect()
}

fn quadrature_with_supporting_points(
    method: QuadratureFormula,
    f: Function1D,
    points: &[f64],
) -> f64 {
    let mut sum = 0.0;
    for i in 0..points.len() - 1 {
        sum += method(f, &Interval::new(points[i], points[i + 1]));
    }
    sum
}

/// Specialty function for task 1.
/// Could be made more general.
pub fn get_convergence_order(run1: &QuadratureTestResult, run2: &QuadratureTestResult) -> f64 {
    (ln!(run2.abs_error) - ln!(run1.abs_error)) / (ln!(run2.h) - ln!(run1.h))
}
