use crate::definitions::{SampleableFunction, InitialValueSystemProblem, Point2D};
use derive_new::*;

pub trait OneStepMethodStep<FT: SampleableFunction<(f64, Vec<f64>), f64>> {
    fn step(&self,
            dfs: &[FT],
            t: f64,
            last_values: &[f64],
            h: f64,
    ) -> Vec<f64>;
}

#[derive(new)]
pub struct OneStepMethod<FT: SampleableFunction<(f64, Vec<f64>), f64>, STEP: OneStepMethodStep<FT>> {
    step_method: STEP,
    ivp: InitialValueSystemProblem<FT>,
    h: f64,
}

impl<FT: SampleableFunction<(f64, Vec<f64>), f64>, STEP: OneStepMethodStep<FT>> OneStepMethod<FT, STEP> {
    pub fn interval(
        &self,
        t_target: f64,
        skip_n: isize,
    ) -> Vec<Vec<Point2D>> {
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
                intermediate_values.push(values.iter().map(|val| Point2D { x: t, y: *val }).collect());
                skip = skip_n
            }
        }

        values = self.step_method.step(&self.ivp.dfs, t, &values, t_target - t);
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

impl<FT: SampleableFunction<(f64, Vec<f64>), f64>, STEP: OneStepMethodStep<FT>> SampleableFunction<f64, Vec<f64>> for OneStepMethod<FT, STEP> {
    fn value_at(&self, t_target: f64) -> Vec<f64> {
        let results = self.interval(t_target, 0);
        results.last().unwrap().iter().map(|p| p.y).collect()
    }
}