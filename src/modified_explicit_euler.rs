use crate::definitions::{InitialValueProblem, DifferentiableFunction, Point2D};

/// Simple implementation of the step taken during the explicit euler function.
fn modified_explicit_euler_step<FT: DifferentiableFunction<(f64, f64)>>(df: &FT, t: f64, last_value: f64, h: f64) -> f64 {
    last_value + h * df.value_at((t, last_value)) + (h * h / 2.0) * df.derivative_at((t, last_value))
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
pub fn modified_explicit_euler<FT: DifferentiableFunction<(f64, f64)>>(ivp: InitialValueProblem<FT>, h: f64, t_target: f64) -> f64 {
    modified_explicit_euler_interval(ivp, h, t_target, 0)
        .last()
        .unwrap()
        .y
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
pub fn modified_explicit_euler_interval<FT: DifferentiableFunction<(f64, f64)>>(
    ivp: InitialValueProblem<FT>,
    h: f64,
    t_target: f64,
    skip_n: isize,
) -> Vec<Point2D> {
    let mut skip: isize = skip_n;
    let mut t = ivp.start_time;
    let mut val = ivp.start_value;
    let mut vals = Vec::with_capacity(((t_target - ivp.start_time) / h).ceil() as usize);
    vals.push(Point2D { x: t, y: val });

    while t + h < t_target {
        val = modified_explicit_euler_step(&ivp.df, t, val, h);
        t += h;
        skip -= 1;
        if skip <= 0 {
            vals.push(Point2D { x: t, y: val });
            skip = skip_n
        }
    }

    val = modified_explicit_euler_step(&ivp.df, t, val, t_target - t);
    vals.push(Point2D {
        x: t_target,
        y: val,
    });
    vals
}