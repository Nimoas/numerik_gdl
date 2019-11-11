use crate::definitions::{
    ComposeSampleableFunction, Function, Function2D, FunctionND, InitialValueProblem,
    InitialValueSystemProblem, Point2D, SampleableFunction, SubSampleableFunction,
};
use crate::util::euclidean_norm;
use crate::{abs, ceil};
use derive_new::*;
use rayon::prelude::*;
use std::f64::EPSILON;

/// Simple implementation of the step taken during the explicit euler function.
/// Done for arbitrary dimensions.
fn explicit_euler_system_step<FT: SampleableFunction<(f64, Vec<f64>), f64>>(
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
    explicit_euler_system_interval(&problem, h, t_target, skip_n)
        .iter()
        .map(|vec_x| vec_x[0])
        .collect()
}

/// Implementation of the explicit euler method for multiple dimensions.
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
/// use ngdl_rust::euler_explicit::explicit_euler_system;
/// use ngdl_rust::definitions::{Function, InitialValueSystemProblem};
///
/// let h = 0.1;
/// let t_target = 1.0;
/// let dfx: Function<(f64, Vec<f64>)> = |(_, v)| -v[0] + v[1];
/// let dfy: Function<(f64, Vec<f64>)> = |(_, v)| v[0] - v[1];
///
/// let problem = InitialValueSystemProblem::new(0.0, vec![1.0, 0.0], vec![dfx, dfy]);
///
/// dbg!(explicit_euler_system(&problem, h, t_target));
/// ```
pub fn explicit_euler_system<FT: SampleableFunction<(f64, Vec<f64>), f64>>(
    ivp: &InitialValueSystemProblem<FT>,
    h: f64,
    t_target: f64,
) -> Vec<f64> {
    let results = explicit_euler_system_interval(ivp, h, t_target, 0);
    results.last().unwrap().iter().map(|p| p.y).collect()
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
pub fn explicit_euler_system_interval<FT: SampleableFunction<(f64, Vec<f64>), f64>>(
    ivp: &InitialValueSystemProblem<FT>,
    h: f64,
    t_target: f64,
    skip_n: isize,
) -> Vec<Vec<Point2D>> {
    let mut skip: isize = skip_n;
    let mut t = ivp.start_time;
    let mut values = ivp.start_values.clone();
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
    intermediate_values.push(
        values
            .iter()
            .map(|val| Point2D {
                x: t_target,
                y: *val,
            })
            .collect(),
    );
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

/// Gets the right sided slope of the calculated polygonal
pub fn get_differential_at<FT: SampleableFunction<(f64, Vec<f64>), f64>>(
    ivp: &InitialValueSystemProblem<FT>,
    h: f64,
    t_target: f64,
) -> Vec<f64> {
    // We go a step farther, because we want the right slope
    let results = explicit_euler_system_interval(&ivp, h, t_target + h, 0);
    // Get the first x coord that is strictly larger than the target
    let idx = if abs!(t_target - ivp.start_time) > EPSILON {
        ceil!((t_target - ivp.start_time) / h)
    } else {
        1
    };
    // Calculate line slope
    results[idx - 1]
        .iter()
        .zip(&results[idx])
        .map(|(p1, p2)| (p2.y - p1.y) / (p2.x - p1.x))
        .collect()
}

/// Convenience wrapper around the euler method.
/// It can be sampled with a t and returns the results obtained for that.
/// Useful if one wants to feed it into another algorithm, e.g. quadrature.
///
/// Note: Sampling this is probably horrible for performance. Too lazy to make it fast.
#[derive(Debug, new)]
pub struct SampleableEulerPolygonal<FT: SampleableFunction<(f64, Vec<f64>), f64>> {
    ivp: InitialValueSystemProblem<FT>,
    h: f64,
}

impl<FT: SampleableFunction<(f64, Vec<f64>), f64>> SampleableFunction<f64, Vec<f64>>
    for SampleableEulerPolygonal<FT>
{
    fn value_at(&self, input: f64) -> Vec<f64> {
        explicit_euler_system(&self.ivp, self.h, input)
    }
}

/// Similar to the SampleableEulerPolygonal, but returns the slope at t.
#[derive(Debug, new)]
pub struct SampleableEulerMethodDifferential<FT: SampleableFunction<(f64, Vec<f64>), f64>> {
    ivp: InitialValueSystemProblem<FT>,
    h: f64,
}

impl<FT: SampleableFunction<(f64, Vec<f64>), f64>> SampleableFunction<f64, Vec<f64>>
    for SampleableEulerMethodDifferential<FT>
{
    fn value_at(&self, input: f64) -> Vec<f64> {
        get_differential_at(&self.ivp, self.h, input)
    }
}

/// Get the residual as defined for the error bounds of the euler method.
/// This returns a function that can be sampled to get the residual at a certain t.
pub fn get_residual_function(
    create_problem: fn() -> InitialValueSystemProblem<Function<(f64, Vec<f64>)>>,
    h: f64,
) -> impl SampleableFunction<f64, f64> {
    // First a priori error
    let slope_polygonal_spline = SampleableEulerMethodDifferential::new(create_problem(), h);
    let polygonal_spline = SampleableEulerPolygonal::new(create_problem(), h);
    // f (spline values)
    let f: FunctionND<Vec<f64>, Vec<f64>> = |vec_x| vec![-vec_x[0] + vec_x[1], vec_x[0] - vec_x[1]];
    let f_tmp = ComposeSampleableFunction::new(polygonal_spline, f);
    let under_norm = SubSampleableFunction::new(slope_polygonal_spline, f_tmp);
    let f: Function<Vec<f64>> = |v| euclidean_norm(v);
    let residual = ComposeSampleableFunction::new(under_norm, f);
    residual
}
