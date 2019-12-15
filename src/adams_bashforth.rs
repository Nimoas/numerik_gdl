use crate::definitions::{InitialValueSystemProblem, PointwiseAdd, SampleableFunction, ScalarMul};
use crate::generalized_explicit_k_step_method::{KStepMethod, KStepMethodStep};
use crate::generalized_explicit_one_step_method::{OneStepMethod, OneStepMethodStep};

/// For now only a simple one like 2
/// Formula for this is y_{n+2} = y_{n+1} + h * (3/2 * f_{n+1} - 1/2 * f_n)
/// Linear method with a_2 = 1, a_1 = -1, a_0 = 0; b_2 = 0, b_1 = 3/2, b_0 = -1/2
/// Important: Does not hit t_target exactly, because we need equidistant supports
pub struct AdamsBashford2;

impl<FT: SampleableFunction<(f64, Vec<f64>), f64>> KStepMethodStep<FT> for AdamsBashford2 {
    fn step(&self, _k: usize, dfs: &[FT], t: f64, last_values: &[Vec<f64>], h: f64) -> Vec<f64> {
        let last_values_owned: Vec<f64> = last_values[1].clone();
        let before_last_values_owned: Vec<f64> = last_values[0].clone();
        dfs.iter()
            .map(|df| {
                1.5 * df.value_at((t, last_values_owned.clone()))
                    - 0.5 * df.value_at((t - h, before_last_values_owned.clone()))
            })
            .collect::<Vec<f64>>()
            .scalar_mul(h)
            .pointwise_add(last_values_owned) // x_{n-1}
    }
}

/// Makes a system of ODEs into a sampleable function using an Adams-Bashforth method.
pub fn make_adams_bashforth_2_method<
    FT: SampleableFunction<(f64, Vec<f64>), f64>,
    StartStep: OneStepMethodStep<FT>,
>(
    ivp: fn() -> InitialValueSystemProblem<FT>,
    h: f64,
    start_method_gen: fn(
        ivp: InitialValueSystemProblem<FT>,
        h: f64,
    ) -> OneStepMethod<FT, StartStep>,
) -> KStepMethod<FT, AdamsBashford2, StartStep> {
    KStepMethod::new(ivp, h, 2, AdamsBashford2, start_method_gen)
}

/// 3rd order Adams Bashford method.
pub struct AdamsBashford3;

impl<FT: SampleableFunction<(f64, Vec<f64>), f64>> KStepMethodStep<FT> for AdamsBashford3 {
    fn step(&self, _k: usize, dfs: &[FT], t: f64, last_values: &[Vec<f64>], h: f64) -> Vec<f64> {
        let last_values_owned: Vec<f64> = last_values[2].clone();
        let before_last_values_owned: Vec<f64> = last_values[1].clone();
        let before_before_last_values_owned: Vec<f64> = last_values[0].clone();
        dfs.iter()
            .map(|df| {
                (23.0 / 12.0) * df.value_at((t, last_values_owned.clone()))
                    - (16.0 / 12.0) * df.value_at((t - h, before_last_values_owned.clone()))
                    + (5.0 / 12.0)
                        * df.value_at((t - 2.0 * h, before_before_last_values_owned.clone()))
            })
            .collect::<Vec<f64>>()
            .scalar_mul(h)
            .pointwise_add(last_values_owned) // x_{n-2}
    }
}

/// Makes a system of ODEs into a sampleable function using an Adams-Bashforth method.
pub fn make_adams_bashforth_3_method<
    FT: SampleableFunction<(f64, Vec<f64>), f64>,
    StartStep: OneStepMethodStep<FT>,
>(
    ivp: fn() -> InitialValueSystemProblem<FT>,
    h: f64,
    start_method_gen: fn(
        ivp: InitialValueSystemProblem<FT>,
        h: f64,
    ) -> OneStepMethod<FT, StartStep>,
) -> KStepMethod<FT, AdamsBashford3, StartStep> {
    KStepMethod::new(ivp, h, 3, AdamsBashford3, start_method_gen)
}
