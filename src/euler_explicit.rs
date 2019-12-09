use crate::definitions::{ComposeSampleableFunction, Function, Function2D, FunctionND, InitialValueProblem, InitialValueSystemProblem, Point2D, PointwiseAdd, SampleableFunction, ScalarMul, SubSampleableFunction, ODEMethod};
use crate::generalized_explicit_one_step_method::{OneStepMethod, OneStepMethodStep};
use crate::util::euclidean_norm;
use rayon::prelude::*;

/// Simple implementation of the explicit euler method.
/// Lands on target even if h does not match.
///
/// # Arguments
///
/// * `ivp` - The initial value problem we want to approximate
/// * `h` - Step size of the algorithm
/// * `t_target` - Target time we want to get the value for.
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
/// * `t_target` - Target time we want to get the value for.
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
/// * `t_target` - Target time we want to get the value for.
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
pub fn explicit_euler_system<FT: SampleableFunction<(f64, Vec<f64>), f64> + Clone>(
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
/// * `t_target` - Target time we want to get the value for.
/// * `skip_n` - If > 0 only returns ever n-th value to reduce memory footprint while retaining smaller h
///
pub fn explicit_euler_system_interval<FT: SampleableFunction<(f64, Vec<f64>), f64> + Clone>(
    ivp: &InitialValueSystemProblem<FT>,
    h: f64,
    t_target: f64,
    skip_n: isize,
) -> Vec<Vec<Point2D>> {
    let method = make_explicit_euler_method_system((*ivp).clone(), h);
    method.interval(t_target, skip_n)
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

/// Get the residual as defined for the error bounds of the euler method.
/// This returns a function that can be sampled to get the residual at a certain t.
pub fn get_residual_function(
    create_problem: fn() -> InitialValueSystemProblem<Function<(f64, Vec<f64>)>>,
    h: f64,
) -> impl SampleableFunction<f64, f64> {
    // First a priori error
    let euler_method = make_explicit_euler_method_system(create_problem(), h);
    let slope_polygonal_spline =
        make_explicit_euler_method_system(create_problem(), h).get_derivative();
    // f (spline values)
    let f: FunctionND<Vec<f64>, Vec<f64>> = |vec_x| vec![-vec_x[0] + vec_x[1], vec_x[0] - vec_x[1]];
    let f_tmp = ComposeSampleableFunction::new(euler_method, f);
    let under_norm = SubSampleableFunction::new(slope_polygonal_spline, f_tmp);
    let f: Function<Vec<f64>> = |v| euclidean_norm(v);
    let residual = ComposeSampleableFunction::new(under_norm, f);
    residual
}

/// Simple implementation of the step taken during the explicit euler function.
/// Done for arbitrary dimensions.
pub struct ExplicitEulerStep;

impl<FT: SampleableFunction<(f64, Vec<f64>), f64>> OneStepMethodStep<FT> for ExplicitEulerStep {
    fn step(&self, dfs: &[FT], t: f64, last_values: &[f64], h: f64) -> Vec<f64> {
        let last_values_owned: Vec<f64> = last_values.to_vec();
        dfs.iter()
            .map(|df| df.value_at((t, last_values_owned.clone())))
            .collect::<Vec<f64>>()
            .scalar_mul(h)
            .pointwise_add(last_values_owned)
    }
}

/// Makes a system of ODEs into a sampleable function using the explicit euler method.
pub fn make_explicit_euler_method_system<FT: SampleableFunction<(f64, Vec<f64>), f64>>(
    ivp: InitialValueSystemProblem<FT>,
    h: f64,
) -> OneStepMethod<FT, ExplicitEulerStep> {
    OneStepMethod::new(ExplicitEulerStep, ivp, h)
}
