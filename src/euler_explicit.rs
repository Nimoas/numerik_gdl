use crate::definitions::{
    Function2D, InitialValueProblem, InitialValueSystemProblem, Point2D, SampleableFunction,
};
use rayon::prelude::*;

/// Simple implementation of the step taken during the explicit euler function.
/// Done for arbitrary dimensions.
fn explicit_euler_system_step<FT: SampleableFunction<(f64, Vec<f64>)>>(
    dfs: &[FT],
    t: f64,
    last_values: &[f64],
    h: f64,
) -> Vec<f64> {
    let last_values_owned: Vec<f64> = last_values.to_vec();
    dfs.iter()
        .zip(last_values)
        .map(|(df, last_value)| last_value + h * df.value_at((t, last_values_owned.clone())))
        .collect()
}

/// Simple implementation of the explicit euler method.
/// Lands on target even if h does not match.
///
/// # Arguments
///
/// * `ivp` - The initial value problem we want to approximate
/// * `h` - Step size of the algorithm
/// * `t_target` - Target time we want to get the value for. Note that this should be directly reachable with t0 + k * h
///
/// # Example
/// ```
/// use ngdl_rust::definitions::{InitialValueProblem, Function2D};
/// use ngdl_rust::euler_explicit::explicit_euler;
///
/// let ivp: InitialValueProblem<Function2D> = InitialValueProblem::new(0.0, 1.0, |(_, x)| x*x);
/// dbg!(explicit_euler(ivp, 0.001, 1.0));
/// ```
pub fn explicit_euler(ivp: InitialValueProblem<Function2D>, h: f64, t_target: f64) -> f64 {
    let ps = explicit_euler_interval(ivp, h, t_target, 0);
    let p = ps.last().unwrap();
    p.y
}

/// Simple implementation of the explicit euler method.
/// Lands on target even if h does not match.
/// The function returns the intermediate values as well as the final value in the form of a point vector
/// Piggy-backs on the multi-dimensional implementation.
///
/// # Arguments
///
/// * `ivp` - The initial value problem we want to approximate
/// * `h` - Step size of the algorithm
/// * `t_target` - Target time we want to get the value for. Note that this should be directly reachable with t0 + k * h
/// * `skip_n` - If > 0 only returns ever n-th value to reduce memory footprint while retaining smaller h
///
pub fn explicit_euler_interval(
    ivp: InitialValueProblem<Function2D>,
    h: f64,
    t_target: f64,
    skip_n: isize,
) -> Vec<Point2D> {
    let problem = ivp.to_system_problem();
    explicit_euler_system_interval(problem, h, t_target, skip_n)
        .iter()
        .map(|vec_x| vec_x[0])
        .collect()
}

/// Implementation of the explicit euler method for multiple dimensions.
/// Lands on target even if h does not match.
/// The function returns the intermediate values as well as the final value in the form of a point vector
///
/// # Arguments
///
/// * `ivp` - The initial value problem we want to approximate
/// * `h` - Step size of the algorithm
/// * `t_target` - Target time we want to get the value for. Note that this should be directly reachable with t0 + k * h
/// * `skip_n` - If > 0 only returns ever n-th value to reduce memory footprint while retaining smaller h
///
pub fn explicit_euler_system_interval<FT: SampleableFunction<(f64, Vec<f64>)>>(
    ivp: InitialValueSystemProblem<FT>,
    h: f64,
    t_target: f64,
    skip_n: isize,
) -> Vec<Vec<Point2D>> {
    let mut skip: isize = skip_n;
    let mut t = ivp.start_time;
    let mut values = ivp.start_values;
    let mut intermediate_values: Vec<Vec<Point2D>> =
        Vec::with_capacity(((t_target - ivp.start_time) / h).ceil() as usize);
    intermediate_values.push(values.iter().map(|val| Point2D { x: t, y: *val }).collect());

    while t + h < t_target {
        values = explicit_euler_system_step(&ivp.dfs, t, &values, h);

        t += h;
        skip -= 1;
        if skip <= 0 {
            intermediate_values.push(values.iter().map(|val| Point2D { x: t, y: *val }).collect());
            skip = skip_n
        }
    }

    values = explicit_euler_system_step(&ivp.dfs, t, &values, t_target - t);
    intermediate_values.push(values.iter().map(|val| Point2D { x: t, y: *val }).collect());
    intermediate_values
}

/// Runs the explicit euler method for all supplied h in parallel.
pub fn explicit_euler_test_run(
    ivp: InitialValueProblem<Function2D>,
    h: &[f64],
    t_target: f64,
) -> Vec<f64> {
    h.par_iter()
        .map(|h| explicit_euler(ivp, *h, t_target))
        .collect()
}

/// Runs the explicit euler method (interval version) for all supplied h in parallel.
pub fn explicit_euler_interval_test_run(
    ivp: InitialValueProblem<Function2D>,
    h: &[f64],
    t_target: f64,
    skip_n: isize,
) -> Vec<Vec<Point2D>> {
    h.par_iter()
        .map(|h| explicit_euler_interval(ivp, *h, t_target, skip_n))
        .collect()
}
