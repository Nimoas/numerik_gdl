use crate::definitions::{
    ClosureDifferentiableFunction2D, InitialValueProblem, Point2D, SimpleDifferentiableFunction2D,
};
use crate::newton_method::newton_method;
use rayon::prelude::*;

pub fn implicit_euler(
    ivp: InitialValueProblem<SimpleDifferentiableFunction2D>,
    h: f64,
    t_target: f64,
) -> f64 {
    implicit_euler_interval(ivp, h, t_target, 0)
        .last()
        .unwrap()
        .y
}

pub fn implicit_euler_interval(
    ivp: InitialValueProblem<SimpleDifferentiableFunction2D>,
    h: f64,
    t_target: f64,
    skip_n: isize,
) -> Vec<Point2D> {
    let mut skip = skip_n;
    let mut t = ivp.start_time;
    let mut val = ivp.start_value;
    let mut vals = Vec::with_capacity(((t_target - ivp.start_time) / h).ceil() as usize);
    vals.push(Point2D { x: t, y: val });

    while t + h < t_target {
        val = implicit_euler_step(ivp.df, t, val, h);
        t += h;
        skip -= 1;
        if skip <= 0 {
            vals.push(Point2D { x: t, y: val });
            //dbg!(Point2D { x: t, y: val });
            skip = skip_n;
        }
    }

    val = implicit_euler_step(ivp.df, t, val, t_target - t);
    vals.push(Point2D {
        x: t_target,
        y: val,
    });
    vals
}

fn implicit_euler_step(df: SimpleDifferentiableFunction2D, t: f64, val: f64, h: f64) -> f64 {
    let to_solve =
        |t: f64, x: f64, (last, delta, g): (f64, f64, SimpleDifferentiableFunction2D)| {
            last + delta * (g.f)(t, x) - x
        };
    let to_solve_derivative =
        |t: f64, x: f64, (_, delta, g): (f64, f64, SimpleDifferentiableFunction2D)| {
            delta * (g.df)(t, x) - 1.0
        };
    let func: ClosureDifferentiableFunction2D<(f64, f64, SimpleDifferentiableFunction2D)> =
        ClosureDifferentiableFunction2D::new((val, h, df), to_solve, to_solve_derivative);
    newton_method(func, t, val, 0.001 * h)
}

pub fn implicit_euler_interval_test_run(
    ivp: InitialValueProblem<SimpleDifferentiableFunction2D>,
    h: &[f64],
    t_target: f64,
    skip_n: isize,
) -> Vec<Vec<Point2D>> {
    h.par_iter()
        .map(|h| implicit_euler_interval(ivp, *h, t_target, skip_n))
        .collect()
}
