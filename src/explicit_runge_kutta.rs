use crate::definitions::{InitialValueSystemProblem, PointwiseAdd, SampleableFunction, ScalarMul};
use crate::generalized_explicit_one_step_method::{OneStepMethod, OneStepMethodStep};
use crate::util::make_zero_vec;
use derive_new::*;
use std::marker::PhantomData;

/// This is a tableau for a Runge-Kutta method.
/// I would prefer to enforce same size for all of these, but the required Rust feature (const generics) has not yet stabilized.
#[derive(Clone, Debug, new)]
pub struct Tableau {
    cs: Vec<f64>,
    bs: Vec<f64>,
    coeffs: Vec<Vec<f64>>,
}

/// Implementation for a RK method with only explicit components.
#[derive(Clone, Debug, new)]
pub struct ExplicitRungeKuttaMethod<FT: SampleableFunction<(f64, Vec<f64>), f64>> {
    _t: PhantomData<FT>,
    tableau: Tableau,
}

impl<FT: SampleableFunction<(f64, Vec<f64>), f64>> ExplicitRungeKuttaMethod<FT> {
    fn get_ks(&self, dfs: &[FT], t: f64, last_values: &[f64], h: f64) -> Vec<Vec<f64>> {
        let mut ks: Vec<Vec<f64>> = Vec::with_capacity(self.tableau.cs.len());

        for (idx, c) in self.tableau.cs.iter().enumerate() {
            // We currently calculate k_idx
            let t_sample = t + h * *c;
            let mut sample_vals: Vec<f64> = make_zero_vec(dfs.len());
            // For the current row take all as that are below the diagonal
            for (idx_inner, a) in self.tableau.coeffs[idx].iter().enumerate().take(idx) {
                // get the values in k_idx_inner, multiply with a and sum up
                sample_vals = sample_vals.pointwise_add(ks[idx_inner].clone().scalar_mul(*a));
            }
            sample_vals = sample_vals
                .scalar_mul(h)
                .pointwise_add(last_values.iter().copied().collect());

            ks.push(
                dfs.iter()
                    .map(|f| f.value_at((t_sample, sample_vals.clone())))
                    .collect(),
            );
        }

        ks
    }
}

impl<FT: SampleableFunction<(f64, Vec<f64>), f64>> OneStepMethodStep<FT>
    for ExplicitRungeKuttaMethod<FT>
{
    fn step(&self, dfs: &[FT], t: f64, last_values: &[f64], h: f64) -> Vec<f64> {
        let ks: Vec<Vec<f64>> = self.get_ks(dfs, t, last_values, h);
        let change_term: Vec<f64> = self
            .tableau
            .bs
            .iter()
            .zip(ks)
            .map(|(b, k)| k.scalar_mul(*b))
            .fold(make_zero_vec(dfs.len()), |v1, v2| v1.pointwise_add(v2));

        change_term
            .scalar_mul(h)
            .pointwise_add(last_values.iter().copied().collect())
    }
}

/// Creates a new Runge-Kutta method for the given system and tableau.
/// Intended to be re-exported via functions with fixed tableaus, e.g. for classical RK.
/// The resulting method can be sampled at any t.
///
/// # Example
/// ```
///    use ngdl_rust::explicit_runge_kutta::{make_explicit_runge_kutta_with_tableau, Tableau};
///    use ngdl_rust::definitions::{Function, InitialValueSystemProblem, ODEMethod};
///
///    let dfr: Function<(f64, Vec<f64>)> = |(_, v)| v[1]*v[0];
///    let dfz: Function<(f64, Vec<f64>)> = |(_, v)| 5.0 * v[1];
///
///    let problem = InitialValueSystemProblem::new(0.0, vec![0.0, -1.0], vec![dfr, dfz]);
///    let tableau = Tableau::new(
///        vec![0.0, 0.5, 1.0, 1.0], // cs
///        vec![1.0 / 6.0, 2.0 / 3.0, 0.0, 1.0 / 6.0], // bs
///        vec![vec![],
///             vec![0.5],
///             vec![0.0, 1.0],
///             vec![0.0, 0.0, 1.0]], //as
///    );
///
///    let rk_method = make_explicit_runge_kutta_with_tableau(problem, 0.1, tableau);
///
///    let approximation = rk_method.interval(3.14, 0);
/// ```
pub fn make_explicit_runge_kutta_with_tableau<FT: SampleableFunction<(f64, Vec<f64>), f64>>(
    ivp: InitialValueSystemProblem<FT>,
    h: f64,
    tableau: Tableau,
) -> OneStepMethod<FT, ExplicitRungeKuttaMethod<FT>> {
    OneStepMethod::new(ExplicitRungeKuttaMethod::new(tableau), ivp, h)
}

/// Classic 4-th order explicit RK method.
pub fn make_classic_runge_kutta<FT: SampleableFunction<(f64, Vec<f64>), f64>>(
    ivp: InitialValueSystemProblem<FT>,
    h: f64,
) -> OneStepMethod<FT, ExplicitRungeKuttaMethod<FT>> {
    let tableau = Tableau::new(
        vec![0.0, 0.5, 0.5, 1.0],                         // cs
        vec![1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0], // bs
        vec![vec![], vec![0.5], vec![0.0, 0.5], vec![0.0, 0.0, 1.0]],
    );

    OneStepMethod::new(ExplicitRungeKuttaMethod::new(tableau), ivp, h)
}

/// England explicit RK method.
pub fn make_england_runge_kutta<FT: SampleableFunction<(f64, Vec<f64>), f64>>(
    ivp: InitialValueSystemProblem<FT>,
    h: f64,
) -> OneStepMethod<FT, ExplicitRungeKuttaMethod<FT>> {
    let tableau = Tableau::new(
        vec![0.0, 0.5, 0.5, 1.0],                   // cs
        vec![1.0 / 6.0, 0.0, 2.0 / 3.0, 1.0 / 6.0], // bs
        vec![vec![], vec![0.5], vec![0.25, 0.25], vec![0.0, -1.0, 2.0]],
    );

    OneStepMethod::new(ExplicitRungeKuttaMethod::new(tableau), ivp, h)
}

/// 3/8 explicit RK method.
pub fn make_three_eight_runge_kutta<FT: SampleableFunction<(f64, Vec<f64>), f64>>(
    ivp: InitialValueSystemProblem<FT>,
    h: f64,
) -> OneStepMethod<FT, ExplicitRungeKuttaMethod<FT>> {
    let tableau = Tableau::new(
        vec![0.0, 0.5, 1.0, 1.0],                   // cs
        vec![1.0 / 6.0, 2.0 / 3.0, 0.0, 1.0 / 6.0], // bs
        vec![vec![], vec![0.5], vec![0.0, 1.0], vec![0.0, 0.0, 1.0]],
    );

    OneStepMethod::new(ExplicitRungeKuttaMethod::new(tableau), ivp, h)
}

/// 2 Step RK Method of order 2.
pub fn make_2nd_order_runge_kutta<FT: SampleableFunction<(f64, Vec<f64>), f64>>(
    ivp: InitialValueSystemProblem<FT>,
    h: f64,
) -> OneStepMethod<FT, ExplicitRungeKuttaMethod<FT>> {
    let tableau = Tableau::new(
        vec![0.0, 0.5], // cs
        vec![0.0, 1.0], // bs
        vec![vec![], vec![0.5]],
    );

    OneStepMethod::new(ExplicitRungeKuttaMethod::new(tableau), ivp, h)
}

/// Heun Method of order 3.
pub fn make_heun_method<FT: SampleableFunction<(f64, Vec<f64>), f64>>(
    ivp: InitialValueSystemProblem<FT>,
    h: f64,
) -> OneStepMethod<FT, ExplicitRungeKuttaMethod<FT>> {
    let tableau = Tableau::new(
        vec![0.0, 1.0 / 3.0, 2.0 / 3.0], // cs
        vec![0.25, 0.0, 0.75],           // bs
        vec![vec![], vec![1.0 / 3.0], vec![0.0, 2.0 / 3.0]],
    );

    OneStepMethod::new(ExplicitRungeKuttaMethod::new(tableau), ivp, h)
}
