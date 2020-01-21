use crate::definitions::{InitialValueSystemProblem, Point2D, SampleableFunction};
use crate::explicit_runge_kutta::{make_explicit_runge_kutta_with_tableau, Tableau};
use crate::{abs, powf};
use derive_new::*;
use std::marker::PhantomData;

const PRINT_NUM: isize = 500;

/// Implementation for an embedded RK method with only explicit components.
/// HACK: Only looks at first component for error
#[derive(Clone, Debug, new)]
pub struct EmbeddedExplicitRungeKuttaMethod<FT: SampleableFunction<(f64, Vec<f64>), f64>> {
    _t: PhantomData<FT>,
    tableau: Tableau,
    tableau_lower: Tableau,
    // the one with the lower order
    lower_order: usize,
    current_h: f64,
    make_ivp: fn() -> InitialValueSystemProblem<FT>,
    tolerance: f64,
}

impl<FT: SampleableFunction<(f64, Vec<f64>), f64>> EmbeddedExplicitRungeKuttaMethod<FT> {
    fn step(&mut self, t: f64, last_values: &[f64]) -> Vec<f64> {
        let rk1 = make_explicit_runge_kutta_with_tableau(
            (self.make_ivp)(),
            self.current_h,
            self.tableau.clone(),
        );
        let val1 = rk1.value_at(t + self.current_h);

        let rk2 = make_explicit_runge_kutta_with_tableau(
            (self.make_ivp)(),
            self.current_h,
            self.tableau_lower.clone(),
        );
        let val2 = rk2.value_at(t + self.current_h);

        let err = val1.iter().zip(val2.iter()).zip(last_values.iter())
            .map(|((v1, v2), l)| abs!(v1 - v2) / (1.0 + abs!(l)))
            .fold(0.0, f64::max);

        self.current_h = 2.0f64.min(
            0.5f64
                .max(0.9 * powf!(self.tolerance / err, 1.0 / (self.lower_order as f64 + 1.0))),
        ) * self.current_h;

        if err <= self.tolerance {
            return val1;
        }
        self.step(t, last_values)
    }

    pub fn interval(&mut self, t_target: f64, skip_n: isize) -> Vec<Vec<Point2D>> {
        let ivp = (self.make_ivp)();
        let mut skip: isize = skip_n;
        let mut print_cnt: isize = PRINT_NUM;
        let mut t = ivp.start_time;
        let mut values = ivp.start_values.clone();
        let mut intermediate_values: Vec<Vec<Point2D>> = Vec::new(); // Can't predict steps due to variability

        intermediate_values.push(values.iter().map(|val| Point2D { x: t, y: *val }).collect());

        // We overshoot, but that can't be helped
        while t < t_target {
            values = self.step(t, &values);

            t += self.current_h;
            skip -= 1;
            if skip <= 0 {
                intermediate_values
                    .push(values.iter().map(|val| Point2D { x: t, y: *val }).collect());
                skip = skip_n
            }

            print_cnt -= 1;
            if print_cnt <= 0 {
                print_cnt = PRINT_NUM;
                println!("Current t: {}\nCurrent h: {:E}\n", t, self.current_h);
            }
        }
        intermediate_values
    }
}

pub fn make_embedded_explicit_runge_kutta_with_tableau<
    FT: SampleableFunction<(f64, Vec<f64>), f64>,
>(
    create_ivp: fn() -> InitialValueSystemProblem<FT>,
    h_start: f64,
    tableau1: Tableau,
    tableau2: Tableau,
    lower_order: usize,
    tolerance: f64,
) -> EmbeddedExplicitRungeKuttaMethod<FT> {
    EmbeddedExplicitRungeKuttaMethod::new(
        tableau1,
        tableau2,
        lower_order,
        h_start,
        create_ivp,
        tolerance,
    )
}

pub fn make_embedded_rk_1st_order<FT: SampleableFunction<(f64, Vec<f64>), f64>>(
    create_ivp: fn() -> InitialValueSystemProblem<FT>,
    h_start: f64,
    tolerance: f64,
) -> EmbeddedExplicitRungeKuttaMethod<FT> {
    make_embedded_explicit_runge_kutta_with_tableau(
        create_ivp,
        h_start,
        Tableau::new(
            vec![0.0, 1.0], // cs
            vec![0.5, 0.5], // bs
            vec![vec![], vec![1.0]],
        ),
        Tableau::new(
            vec![0.0, 1.0], // cs
            vec![1.0, 0.0], // bs
            vec![vec![], vec![1.0]],
        ),
        1,
        tolerance,
    )
}

/// DOPRI5 implementation.
pub fn make_dopri5<FT: SampleableFunction<(f64, Vec<f64>), f64>>(
    create_ivp: fn() -> InitialValueSystemProblem<FT>,
    h_start: f64,
    tolerance: f64,
) -> EmbeddedExplicitRungeKuttaMethod<FT> {
    let cs = vec![
        vec![],
        vec![0.2],
        vec![3.0 / 40.0, 9.0 / 40.0],
        vec![44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0],
        vec![
            19372.0 / 6561.0,
            -25360.0 / 2187.0,
            64448.0 / 6561.0,
            -212.0 / 729.0,
        ],
        vec![
            9017.0 / 3168.0,
            -355.0 / 33.0,
            46732.0 / 5247.0,
            49.0 / 176.0,
            -5103.0 / 18656.0,
        ],
        vec![
            35.0 / 384.0,
            0.0,
            500.0 / 1113.0,
            125.0 / 192.0,
            -2187.0 / 6784.0,
            11.0 / 84.0,
        ],
    ];

    make_embedded_explicit_runge_kutta_with_tableau(
        create_ivp,
        h_start,
        Tableau::new(
            vec![0.0, 0.2, 0.3, 0.8, 8.0 / 9.0, 1.0, 1.0], // cs
            vec![
                35.0 / 384.0,
                0.0,
                500.0 / 1113.0,
                125.0 / 192.0,
                -2187.0 / 6784.0,
                11.0 / 84.0,
                0.0,
            ], // bs
            cs.clone(),
        ),
        Tableau::new(
            vec![0.0, 0.2, 0.3, 0.8, 8.0 / 9.0, 1.0, 1.0], // cs
            vec![
                5179.0 / 57600.0,
                0.0,
                7571.0 / 16695.0,
                393.0 / 640.0,
                -92097.0 / 339200.0,
                187.0 / 2100.0,
                1.0 / 40.0,
            ], // bs
            cs,
        ),
        4,
        tolerance,
    )
}
