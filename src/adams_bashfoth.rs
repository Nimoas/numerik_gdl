use crate::definitions::{SampleableFunction, InitialValueSystemProblem, ODEMethod, Point2D, ScalarMul, PointwiseAdd};
use derive_new::*;
use crate::euler_explicit::make_explicit_euler_method_system;

pub fn adams_bashford_step<FT: SampleableFunction<(f64, Vec<f64>), f64>>(dfs: &[FT], t: f64, values_last: &[f64], values_before_last: &[f64], h: f64) -> Vec<f64> {
    let last_values_owned: Vec<f64> = values_last.to_vec();
    let before_last_values_owned: Vec<f64> = values_before_last.to_vec();
    dfs.iter()
        .map(|df| 1.5 * df.value_at((t, last_values_owned.clone())) - 0.5 * df.value_at((t - h, before_last_values_owned.clone())))
        .collect::<Vec<f64>>()
        .scalar_mul(h)
        .pointwise_add(last_values_owned) // x_{n-1}
}

/// For now only a simple one like 2
/// Formula for this is y_{n+2} = y_{n+1} + h * (3/2 * f_{n+1} - 1/2 * f_n)
/// Linear method with a_2 = 1, a_1 = -1, a_0 = 0; b_2 = 0, b_1 = 3/2, b_0 = -1/2
/// Important: Does not hit t_target exactly, because we need equidistant supports
#[derive(new)]
pub struct AdamsBashforthMethod<FT: SampleableFunction<(f64, Vec<f64>), f64>>
{
    ivp_getter: fn() -> InitialValueSystemProblem<FT>,
    h: f64,
}

impl<FT: SampleableFunction<(f64, Vec<f64>), f64>> ODEMethod for AdamsBashforthMethod<FT> {
    fn interval(&self, t_target: f64, skip_n: isize) -> Vec<Vec<Point2D>> {
        let ivp = (self.ivp_getter)();

        // Bootstrap with Euler method
        // Assume h is always small enough for this
        let euler_method = make_explicit_euler_method_system((self.ivp_getter)(), self.h);
        let x_1 = euler_method.value_at(ivp.start_time + self.h);

        let mut skip: isize = skip_n;
        let mut t = ivp.start_time + self.h;
        let mut before_values = ivp.start_values.clone();
        let mut values = x_1;
        let mut intermediate_values: Vec<Vec<Point2D>> =
            Vec::with_capacity(((t_target - ivp.start_time) / self.h).ceil() as usize);
        intermediate_values.push(before_values.iter().map(|val| Point2D { x: t, y: *val }).collect());
        intermediate_values.push(values.iter().map(|val| Point2D { x: t, y: *val }).collect());

        while t < t_target {
            let tmp = adams_bashford_step(&ivp.dfs, t, &values, &before_values, self.h);
            before_values = values;
            values = tmp;

            t += self.h;
            skip -= 1;
            if skip <= 0 {
                intermediate_values
                    .push(values.iter().map(|val| Point2D { x: t, y: *val }).collect());
                skip = skip_n
            }
        }

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
}

/// Makes a system of ODEs into a sampleable function using an Adams-Bashforth method.
pub fn make_adams_bashforth_method<FT: SampleableFunction<(f64, Vec<f64>), f64>>(
    ivp: fn() -> InitialValueSystemProblem<FT>,
    h: f64,
) -> AdamsBashforthMethod<FT> {
    AdamsBashforthMethod::new(ivp, h)
}