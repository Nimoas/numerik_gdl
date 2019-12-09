use crate::definitions::{
    DifferentiableFunction, InitialValueSystemProblem, ODEMethod, Point2D, SampleableFunction,
    SampledDerivative,
};
use crate::{abs, ceil};
use derive_new::*;
use std::f64::EPSILON;

pub trait OneStepMethodStep<FT: SampleableFunction<(f64, Vec<f64>), f64>> {
    fn step(&self, dfs: &[FT], t: f64, last_values: &[f64], h: f64) -> Vec<f64>;
}

#[derive(new)]
pub struct OneStepMethod<FT: SampleableFunction<(f64, Vec<f64>), f64>, STEP: OneStepMethodStep<FT>>
{
    step_method: STEP,
    ivp: InitialValueSystemProblem<FT>,
    h: f64,
}

impl<FT: SampleableFunction<(f64, Vec<f64>), f64>, STEP: OneStepMethodStep<FT>> ODEMethod
    for OneStepMethod<FT, STEP>
{
    fn interval(&self, t_target: f64, skip_n: isize) -> Vec<Vec<Point2D>> {
        let mut skip: isize = skip_n;
        let mut t = self.ivp.start_time;
        let mut values = self.ivp.start_values.clone();
        let mut intermediate_values: Vec<Vec<Point2D>> =
            Vec::with_capacity(((t_target - self.ivp.start_time) / self.h).ceil() as usize);
        intermediate_values.push(values.iter().map(|val| Point2D { x: t, y: *val }).collect());

        while t + self.h < t_target {
            values = self.step_method.step(&self.ivp.dfs, t, &values, self.h);

            t += self.h;
            skip -= 1;
            if skip <= 0 {
                intermediate_values
                    .push(values.iter().map(|val| Point2D { x: t, y: *val }).collect());
                skip = skip_n
            }
        }

        values = self
            .step_method
            .step(&self.ivp.dfs, t, &values, t_target - t);
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

impl<FT: SampleableFunction<(f64, Vec<f64>), f64>, STEP: OneStepMethodStep<FT>>
    OneStepMethod<FT, STEP>
{
    pub fn get_derivative(self) -> SampledDerivative<f64, Vec<f64>, Self> {
        SampledDerivative::new(self)
    }
}

impl<FT: SampleableFunction<(f64, Vec<f64>), f64>, STEP: OneStepMethodStep<FT>>
    DifferentiableFunction<f64, Vec<f64>> for OneStepMethod<FT, STEP>
{
    fn derivative_at(&self, t_target: f64) -> Vec<f64> {
        // We go a step farther, because we want the right slope
        let results = self.interval(t_target + self.h, 0);
        // Get the first x coord that is strictly larger than the target
        let idx = if abs!(t_target - self.ivp.start_time) > EPSILON {
            ceil!((t_target - self.ivp.start_time) / self.h)
        } else {
            1
        };
        // Calculate line slope
        results[idx - 1]
            .iter()
            .zip(&results[idx])
            .map(|(p1, p2)| (p2.y - p1.y) / (p2.x - p1.x))
            .collect()
    }
}
