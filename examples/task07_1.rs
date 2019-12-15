use gnuplot::Figure;
use gnuplot::PlotOption::{Caption, Color};
use ngdl_rust::definitions::{Function, InitialValueSystemProblem, Interval, ODEMethod, Point2D};
use ngdl_rust::euler_explicit::make_explicit_euler_method_system;
use ngdl_rust::plot_util::{plot_line_on, plot_line_points_on};
use ngdl_rust::util::sample_function_generic;
use ngdl_rust::{cos, exp, sin};
use std::error::Error;
use std::f64::consts::E;
use std::fs::create_dir_all;
use std::ops::Add;

const IMAGE_DIR: &str = "./img_task07_1/";

fn main() -> Result<(), Box<dyn Error>> {
    create_dir_all(IMAGE_DIR)?;

    let h = 0.01;

    let exact_1: fn(f64) -> Point2D = |t| {
        Point2D::new(
            exp!(-t) * (2.0 * sin!(5.0 * t) + cos!(5.0 * t)),
            exp!(-t) * (2.0 * cos!(5.0 * t) - sin!(5.0 * t)),
        )
    };

    let exact_points_1 = sample_function_generic(exact_1, Interval::new(0.0, 4.0), 10000);

    let exact_2: fn(f64) -> Point2D = |t| {
        Point2D::new(
            cos!(2.0 * t) + sin!(2.0 * t),
            2.0 * cos!(2.0 * t) - 2.0 * sin!(2.0 * t),
        )
    };

    let exact_points_2 = sample_function_generic(exact_2, Interval::new(0.0, 4.0), 10000);

    evaluate("problem_1", h, 4.0, create_problem_1(), exact_points_1);
    evaluate("problem_2", h, 15.0, create_problem_2(), exact_points_2);

    Ok(())
}

fn create_problem_1() -> InitialValueSystemProblem<Function<(f64, Vec<f64>)>> {
    let dfx: Function<(f64, Vec<f64>)> = |(_t, v)| -v[0] + 5.0 * v[1];
    let dfy: Function<(f64, Vec<f64>)> = |(_t, v)| -5.0 * v[0] - v[1];
    InitialValueSystemProblem::new(0.0, vec![1.0, 2.0], vec![dfx, dfy])
}

fn create_problem_2() -> InitialValueSystemProblem<Function<(f64, Vec<f64>)>> {
    let dfx: Function<(f64, Vec<f64>)> = |(_t, v)| v[1];
    let dfy: Function<(f64, Vec<f64>)> = |(_t, v)| -4.0 * v[0];
    InitialValueSystemProblem::new(0.0, vec![1.0, 2.0], vec![dfx, dfy])
}

fn evaluate(
    name: &str,
    h: f64,
    target: f64,
    prob: InitialValueSystemProblem<Function<(f64, Vec<f64>)>>,
    exact: Vec<Point2D>,
) {
    let euler_method = make_explicit_euler_method_system(prob, h);
    let approximation = euler_method.interval(target, 0);

    let xs: Vec<f64> = approximation.iter().map(|v| v[0].y).collect();
    let ys: Vec<f64> = approximation.iter().map(|v| v[1].y).collect();

    let to_plot: Vec<_> = xs
        .iter()
        .zip(ys)
        .map(|(x, y)| Point2D::new(*x, y))
        .collect();

    let mut fg = Figure::new();
    let axis = fg.axes2d();
    plot_line_on(axis, &exact, &[Caption("Exact solution"), Color("black")]);
    plot_line_points_on(
        axis,
        &to_plot,
        &[
            Caption(&format!("Euler approximation with h={}", h)),
            Color("red"),
        ],
    );

    let filename = IMAGE_DIR.to_owned().add(name).add(".png");
    fg.save_to_png(&filename, 1200, 800)
        .expect("Unable to save file");
}
