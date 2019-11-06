use crate::definitions::{
    ClosureDifferentiableFunction, InitialValueProblem, Point2D, SimpleDifferentiableFunction,
};
use crate::newton_method::newton_method;
use rayon::prelude::*;

/// Implementation of the implicit euler method.
/// (Not simple because I'm not 100% happy with it.)
/// Lands on target even if h does not match.
/// Uses Newton's method internally.
///
/// # Arguments
///
/// * `ivp` - The initial value problem we want to approximate
/// * `h` - Step size of the algorithm
/// * `t_target` - Target time we want to get the value for. Note that this should be directly reachable with t0 + k * h
///
/// # Example
/// ```
/// use ngdl_rust::definitions::{InitialValueProblem, SimpleDifferentiableFunction};
/// use ngdl_rust::implicit_euler::implicit_euler;
///
/// let ivp: InitialValueProblem<SimpleDifferentiableFunction<(f64, f64)>> =
///        InitialValueProblem::new(
///            0.0,
///            1.0,
///            SimpleDifferentiableFunction::new(
///                |(_, x)| -1000.0 * x,
///                |(_, _)| -1000.0,
///            ),
///        );
/// dbg!(implicit_euler(ivp, 0.001, 1.0));
/// ```
pub fn implicit_euler(
    ivp: InitialValueProblem<SimpleDifferentiableFunction<(f64, f64)>>,
    h: f64,
    t_target: f64,
) -> f64 {
    implicit_euler_interval(ivp, h, t_target, 0)
        .last()
        .unwrap()
        .y
}

/// Implementation of the explicit euler method.
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
pub fn implicit_euler_interval(
    ivp: InitialValueProblem<SimpleDifferentiableFunction<(f64, f64)>>,
    h: f64,
    t_target: f64,
    skip_n: isize,
) -> Vec<Point2D> {
    let mut skip = skip_n;
    let mut t = ivp.start_time;
    let mut val = ivp.start_value;
    let mut vals = Vec::with_capacity(((t_target - ivp.start_time) / h).ceil() as usize);
    vals.push(Point2D { x: t, y: val });

    while t + h < t_target {
        val = implicit_euler_step(ivp.df, t, val, h);
        t += h;
        skip -= 1;
        if skip <= 0 {
            vals.push(Point2D { x: t, y: val });
            //dbg!(Point2D { x: t, y: val });
            skip = skip_n;
        }
    }

    val = implicit_euler_step(ivp.df, t, val, t_target - t);
    vals.push(Point2D {
        x: t_target,
        y: val,
    });
    vals
}

fn implicit_euler_step(
    df: SimpleDifferentiableFunction<(f64, f64)>,
    t: f64,
    val: f64,
    h: f64,
) -> f64 {
    let to_solve =
        |(t, x), (last, delta, g): (f64, f64, SimpleDifferentiableFunction<(f64, f64)>)| {
            last + delta * (g.f)((t + delta, x)) - x
        };
    let to_solve_derivative =
        |(t, x), (_, delta, g): (f64, f64, SimpleDifferentiableFunction<(f64, f64)>)| {
            delta * (g.df)((t + delta, x)) - 1.0
        };
    let func: ClosureDifferentiableFunction<
        (f64, f64),
        (f64, f64, SimpleDifferentiableFunction<(f64, f64)>),
    > = ClosureDifferentiableFunction::new((val, h, df), to_solve, to_solve_derivative);
    newton_method(func, t, val, 0.001 * h)
}

/// Runs the implicit euler method (interval version) for all supplied h in parallel.
pub fn implicit_euler_interval_test_run(
    ivp: InitialValueProblem<SimpleDifferentiableFunction<(f64, f64)>>,
    h: &[f64],
    t_target: f64,
    skip_n: isize,
) -> Vec<Vec<Point2D>> {
    h.par_iter()
        .map(|h| implicit_euler_interval(ivp, *h, t_target, skip_n))
        .collect()
}
