use crate::definitions::{InitialValueSystemProblem, PointwiseAdd, SampleableFunction, ScalarMul};
use crate::generalized_explicit_k_step_method::{KStepMethod, KStepMethodStep};
use crate::generalized_explicit_one_step_method::{OneStepMethod, OneStepMethodStep};

/// 3rd order Nyström method.
pub struct Nystroem3;

impl<FT: SampleableFunction<(f64, Vec<f64>), f64>> KStepMethodStep<FT> for Nystroem3 {
    fn step(&self, _k: usize, dfs: &[FT], t: f64, last_values: &[Vec<f64>], h: f64) -> Vec<f64> {
        let last_values_owned: Vec<f64> = last_values[2].clone();
        let before_last_values_owned: Vec<f64> = last_values[1].clone();
        let before_before_last_values_owned: Vec<f64> = last_values[0].clone();
        dfs.iter()
            .map(|df| {
                (7.0 / 3.0) * df.value_at((t, last_values_owned.clone()))
                    - (2.0 / 3.0) * df.value_at((t - h, before_last_values_owned.clone()))
                    + (1.0 / 3.0)
                        * df.value_at((t - 2.0 * h, before_before_last_values_owned.clone()))
            })
            .collect::<Vec<f64>>()
            .scalar_mul(h)
            .pointwise_add(before_last_values_owned) // x_{n-1}
    }
}

/// Makes a system of ODEs into a sampleable function using an 3rd order Nyström method.
pub fn make_nystroem_3_method<
    FT: SampleableFunction<(f64, Vec<f64>), f64>,
    StartStep: OneStepMethodStep<FT>,
>(
    ivp: fn() -> InitialValueSystemProblem<FT>,
    h: f64,
    start_method_gen: fn(
        ivp: InitialValueSystemProblem<FT>,
        h: f64,
    ) -> OneStepMethod<FT, StartStep>,
) -> KStepMethod<FT, Nystroem3, StartStep> {
    KStepMethod::new(ivp, h, 3, Nystroem3, start_method_gen)
}
