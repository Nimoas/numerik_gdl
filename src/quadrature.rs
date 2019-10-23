use crate::definitions::{Function1D, Interval};
use rayon::prelude::*;

#[derive(Debug, Copy, Clone)]
pub struct QuadratureTestResult {
    pub value: f64,
    pub abs_error: f64,
    pub splits_n: usize,
    pub h: f64,
}

pub type QuadratureFormula = fn(f: Function1D, interval: &Interval) -> f64;

pub fn trapezoid_formula(f: Function1D, interval: &Interval) -> f64 {
    (interval.span() / 2.0) * (f(interval.start()) + f(interval.end()))
}

pub fn kepler_formula(f: Function1D, interval: &Interval) -> f64 {
    let mid = interval.start() + (1.0 / 2.0) * (interval.end() - interval.start());
    (interval.span() / 6.0) * (f(interval.start()) + 4.0 * f(mid) + f(interval.end()))
}

pub fn newton_three_eight_formula(f: Function1D, interval: &Interval) -> f64 {
    let mid1 = interval.start() + (1.0 / 3.0) * (interval.end() - interval.start());
    let mid2 = interval.start() + (2.0 / 3.0) * (interval.end() - interval.start());
    (interval.span() / 8.0) * (f(interval.start()) + 3.0 * f(mid1) + 3.0 * f(mid2) + f(interval.end()))
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

pub fn quadrature(method: QuadratureFormula, f: Function1D, interval: Interval, n_splits: usize) -> f64 {
    let pts = make_supporting_points(n_splits, interval);
    quadrature_with_supporting_points(method, f, &pts)
}

pub fn quadrature_test_run(method: QuadratureFormula, f: Function1D, exact: f64, interval: Interval, up_to_splits: usize) -> Vec<QuadratureTestResult> {
    (1..=up_to_splits).into_par_iter().map(|n| {
        let value = quadrature(method, f, interval, n);
        QuadratureTestResult {
            value,
            abs_error: (exact - value).abs(),
            splits_n: n,
            h: interval.span() / n as f64,
        }
    }).collect()
}

fn quadrature_with_supporting_points(method: QuadratureFormula, f: Function1D, points: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..points.len() - 1 {
        sum += method(f, &Interval::new(points[i], points[i + 1]));
    }
    sum
}

fn make_supporting_points(n: usize, interval: Interval) -> Vec<f64> {
    let h = interval.span() / n as f64;
    let mut pts: Vec<f64> = (0..n).map(|step| interval.start() + step as f64 * h).collect();
    pts.push(interval.end());
    pts
}

pub fn get_convergence_order(run1: &QuadratureTestResult, run2: &QuadratureTestResult) -> f64 {
    (run2.abs_error.ln() - run1.abs_error.ln()) / (run2.h.ln() - run1.h.ln())
}