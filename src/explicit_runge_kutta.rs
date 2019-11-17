use derive_new::*;
use crate::definitions::{SampleableFunction, InitialValueSystemProblem};
use crate::generalized_explicit_one_step_method::{OneStepMethodStep, OneStepMethod};
use std::marker::PhantomData;

#[derive(Clone, Debug, new)]
pub struct Tableau {
    cs: Vec<f64>,
    bs: Vec<f64>,
    coeffs: Vec<Vec<f64>>,
}

#[derive(Clone, Debug, new)]
pub struct ExplicitRungeKuttaMethod<FT: SampleableFunction<(f64, Vec<f64>), f64>> {
    _t: PhantomData<FT>,
    tableau: Tableau
}

impl<FT: SampleableFunction<(f64, Vec<f64>), f64>> ExplicitRungeKuttaMethod<FT> {
    fn get_ks(&self, dfs: &[FT], t: f64, last_values: &[f64], h: f64) -> Vec<Vec<f64>> {
        let mut ks: Vec<Vec<f64>> = Vec::with_capacity(self.tableau.cs.len());

        for (idx, c) in self.tableau.cs.iter().enumerate() {
            //dbg!(idx);
            // We currently calculate k_idx
            let t_sample = t + h * *c;
            let mut sample_vals: Vec<f64> = (0..dfs.len()).map(|_| 0.0).collect();
            //dbg!(&sample_vals);
            // For the current row take all as that are below the diagonal
            for (idx_inner, a) in self.tableau.coeffs[idx].iter().enumerate().take(idx) {
                //dbg!(idx_inner);
                //dbg!(a);
                // get the values in k_idx_inner, multiply with a and sum up
                sample_vals = sample_vals.iter()
                    .zip(ks[idx_inner].clone().iter().map(|k| a * k).collect::<Vec<f64>>())
                    .map(|(x1, x2)| x1 + x2).collect();
                //dbg!(&sample_vals);
            }
            sample_vals = sample_vals.iter().zip(last_values.clone()).map(|(x, last)| last + x * h).collect();

            //dbg!(t_sample);
            //dbg!(&sample_vals);
            //dbg!(&ks);
            ks.push(dfs.iter().map(|f| f.value_at((t_sample, sample_vals.clone()))).collect());
        }

        ks
    }
}

impl<FT: SampleableFunction<(f64, Vec<f64>), f64>> OneStepMethodStep<FT> for ExplicitRungeKuttaMethod<FT> {
    fn step(&self, dfs: &[FT], t: f64, last_values: &[f64], h: f64) -> Vec<f64> {
        let ks: Vec<Vec<f64>> = self.get_ks(dfs, t, last_values, h);
        //dbg!(t);
        //dbg!(&last_values);
        //dbg!(&ks);
        let change_term: Vec<f64> = self.tableau
            .bs.iter()
            .zip(ks)
            .map(|(b, k)| k.iter().map(|kv| b * kv).collect::<Vec<f64>>())
            .fold(
                (0..dfs.len()).map(|_| 0.0).collect(),
                |v1, v2| v1.iter().zip(v2).map(|(val1, val2)| val1 + val2).collect(),
            );

        //dbg!(last_values);
        //dbg!(h);
        //dbg!(&change_term);
        //dbg!(&change_term);
        let result = last_values.iter().zip(change_term.iter().map(|x| x * h)).map(|(x, last)| last + x).collect();

        //dbg!(&result);
        result
    }
}

pub fn make_explicit_runge_kutta_with_tableau<FT: SampleableFunction<(f64, Vec<f64>), f64>>(
    ivp: InitialValueSystemProblem<FT>,
    h: f64,
    tableau: Tableau,
) -> OneStepMethod<FT, ExplicitRungeKuttaMethod<FT>> {
    OneStepMethod::new(ExplicitRungeKuttaMethod::new(tableau), ivp, h)
}