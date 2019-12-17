use crate::definitions::{InitialValueSystemProblem, ODEMethod, Point2D, SampleableFunction};
use crate::generalized_explicit_one_step_method::{OneStepMethod, OneStepMethodStep};
use derive_new::*;

pub trait KStepMethodStep<FT: SampleableFunction<(f64, Vec<f64>), f64>> {
    fn step(&self, k: usize, dfs: &[FT], t: f64, last_values: &[Vec<f64>], h: f64) -> Vec<f64>;
}

#[derive(new)]
pub struct KStepMethod<
    FT: SampleableFunction<(f64, Vec<f64>), f64>,
    STEP: KStepMethodStep<FT>,
    StartStep: OneStepMethodStep<FT>,
> {
    ivp_getter: fn() -> InitialValueSystemProblem<FT>,
    h: f64,
    k: usize,
    step_method: STEP,
    start_method_gen:
        fn(ivp: InitialValueSystemProblem<FT>, h: f64) -> OneStepMethod<FT, StartStep>,
}

impl<
        FT: SampleableFunction<(f64, Vec<f64>), f64>,
        STEP: KStepMethodStep<FT>,
        StartStep: OneStepMethodStep<FT>,
    > ODEMethod for KStepMethod<FT, STEP, StartStep>
{
    fn interval(&self, t_target: f64, skip_n: isize) -> Vec<Vec<Point2D>> {
        let ivp = (self.ivp_getter)();

        // Bootstrap with start method (k-1) values
        // Assume h is always small enough for this
        let start_method = (self.start_method_gen)((self.ivp_getter)(), self.h);
        let mut current_values: Vec<Vec<f64>> = vec![ivp.start_values];
        // k not in the range
        for idx in 1..self.k {
            current_values.push(start_method.value_at(ivp.start_time + idx as f64 * self.h));
        }

        let mut skip: isize = skip_n;
        let mut t = ivp.start_time + self.k as f64 * self.h;

        let mut intermediate_values: Vec<Vec<Point2D>> =
            Vec::with_capacity(((t_target - ivp.start_time) / self.h).ceil() as usize);

        // No skipping for the start values
        current_values.clone().iter().for_each(|v| {
            intermediate_values.push(v.iter().map(|val| Point2D { x: t, y: *val }).collect())
        });

        while t < t_target {
            let tmp = self
                .step_method
                .step(self.k, &ivp.dfs, t, &current_values, self.h);
            current_values.remove(0);
            current_values.push(tmp);

            t += self.h;
            skip -= 1;
            if skip <= 0 {
                intermediate_values.push(
                    current_values
                        .last()
                        .unwrap()
                        .iter()
                        .map(|val| Point2D { x: t, y: *val })
                        .collect(),
                );
                skip = skip_n
            }
        }

        intermediate_values.push(
            current_values
                .last()
                .unwrap()
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
