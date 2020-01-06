use crate::definitions::{InitialValueSystemProblem, SampleableFunction};
use crate::generalized_explicit_k_step_method::{KStepMethod, KStepMethodStep};
use crate::generalized_explicit_one_step_method::{OneStepMethod, OneStepMethodStep};
use crate::{cos, exp};
use std::f64::consts::E;

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

/// Task 10, 3 Method, takes alpha
#[allow(non_camel_case_types)]
pub struct Task_10_3_Method(f64);

impl<FT: SampleableFunction<(f64, Vec<f64>), f64>> KStepMethodStep<FT> for Task_10_3_Method {
    fn step(&self, _k: usize, _dfs: &[FT], t: f64, last_values: &[Vec<f64>], h: f64) -> Vec<f64> {
        let xplus1 = last_values[1][0];
        let x = last_values[0][0];

        let tplus2 = t + h;
        let tplus1 = t;
        let tplus0 = t - h;

        let a_1 = -self.0 - 1.0;
        let a_0 = self.0;
        let b_2 = self.0 / 12.0 + 5.0 / 12.0;
        let b_1 = -2.0 * self.0 / 3.0 + 2.0 / 3.0;
        let b_0 = -5.0 * self.0 / 12.0 - 1.0 / 12.0;

        vec![
            (h * (b_2 * exp!(-tplus2) * cos!(tplus2)
                + b_1 * (-xplus1 + exp!(-tplus1) * cos!(tplus1))
                + b_0 * (-x + exp!(-tplus0) * cos!(tplus0)))
                - a_1 * xplus1
                - a_0 * x)
                / (1.0 + h * b_2),
        ]
    }
}

/// Makes a system of ODEs into a sampleable function.
pub fn make_task_10_3_method<
    FT: SampleableFunction<(f64, Vec<f64>), f64>,
    StartStep: OneStepMethodStep<FT>,
>(
    ivp: fn() -> InitialValueSystemProblem<FT>,
    h: f64,
    start_method_gen: fn(
        ivp: InitialValueSystemProblem<FT>,
        h: f64,
    ) -> OneStepMethod<FT, StartStep>,
    a: f64,
) -> KStepMethod<FT, Task_10_3_Method, StartStep> {
    KStepMethod::new(ivp, h, 2, Task_10_3_Method(a), start_method_gen)
}
