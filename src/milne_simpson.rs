use crate::definitions::{InitialValueSystemProblem, PointwiseAdd, SampleableFunction, ScalarMul};
use crate::generalized_explicit_k_step_method::{KStepMethod, KStepMethodStep};
use crate::generalized_explicit_one_step_method::{OneStepMethod, OneStepMethodStep};

/// Implementation of the Milne Simpson predictor-corrector method.
pub struct MilneSimpson;

impl<FT: SampleableFunction<(f64, Vec<f64>), f64>> KStepMethodStep<FT> for MilneSimpson {
    fn step(&self, _k: usize, dfs: &[FT], t: f64, last_values: &[Vec<f64>], h: f64) -> Vec<f64> {
        // Predictor
        let p = predict(dfs, t, last_values, h);
        // Corrector
        correct(dfs, t, last_values, h, p)
    }
}

fn correct<FT: SampleableFunction<(f64, Vec<f64>), f64>>(
    dfs: &[FT],
    t: f64,
    last_values: &[Vec<f64>],
    h: f64,
    predictor: Vec<f64>,
) -> Vec<f64> {
    let nplus3: Vec<f64> = last_values[3].clone();
    let nplus2: Vec<f64> = last_values[2].clone();

    dfs.iter()
        .map(|df| {
            df.value_at((t - h, nplus2.clone()))
                + 4.0 * df.value_at((t, nplus3.clone()))
                + df.value_at((t + h, predictor.clone()))
        })
        .collect::<Vec<f64>>()
        .scalar_mul(h / 3.0)
        .pointwise_add(nplus2)
}

fn predict<FT: SampleableFunction<(f64, Vec<f64>), f64>>(
    dfs: &[FT],
    t: f64,
    last_values: &[Vec<f64>],
    h: f64,
) -> Vec<f64> {
    let nplus3: Vec<f64> = last_values[3].clone();
    let nplus2: Vec<f64> = last_values[2].clone();
    let nplus1: Vec<f64> = last_values[1].clone();
    let n: Vec<f64> = last_values[0].clone();

    dfs.iter()
        .map(|df| {
            2.0 * df.value_at((t - 2.0 * h, nplus1.clone())) - df.value_at((t - h, nplus2.clone()))
                + 2.0 * df.value_at((t, nplus3.clone()))
        })
        .collect::<Vec<f64>>()
        .scalar_mul(4.0 * h / 3.0)
        .pointwise_add(n)
}

/// Makes a system of ODEs into a sampleable function using an method.
pub fn make_milne_simpson_method<
    FT: SampleableFunction<(f64, Vec<f64>), f64>,
    StartStep: OneStepMethodStep<FT>,
>(
    ivp: fn() -> InitialValueSystemProblem<FT>,
    h: f64,
    start_method_gen: fn(
        ivp: InitialValueSystemProblem<FT>,
        h: f64,
    ) -> OneStepMethod<FT, StartStep>,
) -> KStepMethod<FT, MilneSimpson, StartStep> {
    KStepMethod::new(ivp, h, 4, MilneSimpson, start_method_gen)
}
