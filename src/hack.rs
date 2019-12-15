use crate::definitions::{InitialValueSystemProblem, SampleableFunction};
use crate::generalized_explicit_k_step_method::{KStepMethod, KStepMethodStep};
use crate::generalized_explicit_one_step_method::{OneStepMethod, OneStepMethodStep};

/// Hacked together explicit version of AM for task 9, 1
pub struct AdamsMoultonHack;

impl<FT: SampleableFunction<(f64, Vec<f64>), f64>> KStepMethodStep<FT> for AdamsMoultonHack {
    fn step(&self, _k: usize, _dfs: &[FT], t: f64, last_values: &[Vec<f64>], h: f64) -> Vec<f64> {
        let nplus2 = last_values[2][0];
        let nplus1 = last_values[1][0];
        let n = last_values[0][0];

        let tplus3 = t + h;
        let tplus2 = t;
        let tplus1 = t - h;
        let tplus0 = t - 2.0 * h;

        vec![
            (nplus2
                + (h / 24.0)
                    * (-19.0 * tplus2 * tplus2 * nplus2 + 5.0 * tplus1 * tplus1 * nplus1
                        - tplus0 * tplus0 * n))
                / (1.0 + h * 9.0 * tplus3 * tplus3 / 24.0),
        ]
    }
}

/// Makes a system of ODEs into a sampleable function using an Adams-Bashforth method.
pub fn make_adams_moulton_hack_method<
    FT: SampleableFunction<(f64, Vec<f64>), f64>,
    StartStep: OneStepMethodStep<FT>,
>(
    ivp: fn() -> InitialValueSystemProblem<FT>,
    h: f64,
    start_method_gen: fn(
        ivp: InitialValueSystemProblem<FT>,
        h: f64,
    ) -> OneStepMethod<FT, StartStep>,
) -> KStepMethod<FT, AdamsMoultonHack, StartStep> {
    KStepMethod::new(ivp, h, 3, AdamsMoultonHack, start_method_gen)
}

/// Hacked together explicit version of AM for task 9, 1
pub struct MilneSimpsonHack;

impl<FT: SampleableFunction<(f64, Vec<f64>), f64>> KStepMethodStep<FT> for MilneSimpsonHack {
    fn step(&self, _k: usize, _dfs: &[FT], t: f64, last_values: &[Vec<f64>], h: f64) -> Vec<f64> {
        let nplus3 = last_values[3][0];
        let nplus2 = last_values[2][0];

        let tplus4 = t + h;
        let tplus3 = t;
        let tplus2 = t - h;

        vec![
            (nplus2 - (h / 3.0) * (tplus2 * tplus2 * nplus2 + 4.0 * tplus3 * tplus3 * nplus3))
                / (1.0 + h * tplus4 * tplus4 / 3.0),
        ]
    }
}

/// Makes a system of ODEs into a sampleable function using an Adams-Bashforth method.
pub fn make_milne_simpson_hack_method<
    FT: SampleableFunction<(f64, Vec<f64>), f64>,
    StartStep: OneStepMethodStep<FT>,
>(
    ivp: fn() -> InitialValueSystemProblem<FT>,
    h: f64,
    start_method_gen: fn(
        ivp: InitialValueSystemProblem<FT>,
        h: f64,
    ) -> OneStepMethod<FT, StartStep>,
) -> KStepMethod<FT, MilneSimpsonHack, StartStep> {
    KStepMethod::new(ivp, h, 4, MilneSimpsonHack, start_method_gen)
}
