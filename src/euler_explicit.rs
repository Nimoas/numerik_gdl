use crate::definitions::{Function2D, InitialValueProblem};
use rayon::prelude::*;

/// Simple implementation of the step taken during the explicit euler function.
fn explicit_euler_step(df: Function2D, t: f64, last_value: f64, h: f64) -> f64 {
    last_value + h * df(t, last_value)
}

/// Simple implementation of the explicit euler method.
/// Explicitly ignore the possibility of h not leading to t_target for now.
/// I'll implement a fix for that once I actually need it...
///
/// # Arguments
///
/// * `ivp` - The initial value problem we want to approximate
/// * `h` - Step size of the algorithm
/// * `t_target` - Target time we want to get the value for. Note that this should be directly reachable with t0 + k * h
///
/// # Example
/// ```
/// use ngdl_rust::definitions::InitialValueProblem;
/// use ngdl_rust::euler_explicit::explicit_euler;
///
/// let ivp = InitialValueProblem::new(0.0, 1.0, |_, x| x*x);
/// dbg!(explicit_euler(ivp, 0.001, 1.0));
/// ```
pub fn explicit_euler(ivp: InitialValueProblem, h: f64, t_target: f64) -> f64 {
    let mut t = ivp.start_time;
    let mut val = ivp.start_value;

    while t < t_target {
        val = explicit_euler_step(ivp.df, t, val, h);
        t += h;
    }

    val
}

/// Runs the explicit euler method for all supplied h in parallel.
pub fn explicit_euler_test_run(ivp: InitialValueProblem, h: &[f64], t_target: f64) -> Vec<f64> {
    h.par_iter()
        .map(|h| explicit_euler(ivp, *h, t_target))
        .collect()
}
