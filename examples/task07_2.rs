use gnuplot::Figure;
use gnuplot::PlotOption::{Caption, Color};
use ngdl_rust::definitions::{InitialValueSystemProblem, Point2D, Function, Interval};
use ngdl_rust::plot_util::{plot_line_on, plot_line_points_on};
use ngdl_rust::{cos, sin, exp};
use std::error::Error;
use std::fs::create_dir_all;
use std::ops::Add;
use ngdl_rust::util::{sample_function_generic, sse};
use std::f64::consts::E;
use ngdl_rust::explicit_runge_kutta::{make_classic_runge_kutta, make_england_runge_kutta, make_three_eight_runge_kutta};

const IMAGE_DIR: &str = "./img_task07_2/";

fn main() -> Result<(), Box<dyn Error>> {
    create_dir_all(IMAGE_DIR)?;

    let h1 = 0.01;
    let h2 = 0.1;

    let exact_1: fn(f64) -> Point2D = |t| Point2D::new(
        exp!(-t) * (2.0 * sin!(5.0*t) + cos!(5.0*t)),
        exp!(-t) * (2.0 * cos!(5.0*t) - sin!(5.0*t)),
    );

    let exact_points_1 = sample_function_generic(exact_1, Interval::new(0.0, 4.0), 400);

    let exact_2: fn(f64) -> Point2D = |t| Point2D::new(
        cos!(2.0*t) + sin!(2.0*t),
        2.0 * cos!(2.0*t) - 2.0 * sin!(2.0*t),
    );

    let exact_points_2 = sample_function_generic(exact_2, Interval::new(0.0, 4.0), 150);

    evaluate("problem_1", h1, 4.0, create_problem_1, exact_points_1);
    evaluate("problem_2", h2, 15.0, create_problem_2, exact_points_2);

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

fn evaluate(name: &str, h: f64, target: f64, prob_fn: fn() -> InitialValueSystemProblem<Function<(f64, Vec<f64>)>>, exact: Vec<Point2D>) {
    let classic_rk_method = make_classic_runge_kutta(prob_fn(), h);
    let approximation_classic = classic_rk_method.interval(target, 0);

    let xs_classic: Vec<f64> = approximation_classic.iter().map(|v| v[0].y).collect();
    let ys_classic: Vec<f64> = approximation_classic.iter().map(|v| v[1].y).collect();

    let to_plot_classic: Vec<_> = xs_classic
        .iter()
        .zip(ys_classic)
        .map(|(x, y)| Point2D::new(*x, y))
        .collect();

    println!("SSE for {} with classic ERK method: {:e}", name, sse(&exact, &to_plot_classic));
    println!("MSE for {} with classic ERK method: {:e}\n", name, sse(&exact, &to_plot_classic)/exact.len() as f64);

    let england_rk_method = make_england_runge_kutta(prob_fn(), h);
    let approximation_england = england_rk_method.interval(target, 0);

    let xs_england: Vec<f64> = approximation_england.iter().map(|v| v[0].y).collect();
    let ys_england: Vec<f64> = approximation_england.iter().map(|v| v[1].y).collect();

    let to_plot_england: Vec<_> = xs_england
        .iter()
        .zip(ys_england)
        .map(|(x, y)| Point2D::new(*x, y))
        .collect();

    println!("SSE for {} with England ERK method: {:e}", name, sse(&exact, &to_plot_england));
    println!("MSE for {} with England ERK method: {:e}\n", name, sse(&exact, &to_plot_england)/exact.len() as f64);

    let three_eight_rk_method = make_three_eight_runge_kutta(prob_fn(), h);
    let approximation_three_eight = three_eight_rk_method.interval(target, 0);

    let xs_three_eight: Vec<f64> = approximation_three_eight.iter().map(|v| v[0].y).collect();
    let ys_three_eight: Vec<f64> = approximation_three_eight.iter().map(|v| v[1].y).collect();

    let to_plot_three_eight: Vec<_> = xs_three_eight
        .iter()
        .zip(ys_three_eight)
        .map(|(x, y)| Point2D::new(*x, y))
        .collect();

    println!("SSE for {} with 3/8 ERK method: {:e}", name, sse(&exact, &to_plot_three_eight));
    println!("MSE for {} with 3/8 ERK method: {:e}\n\n\n", name, sse(&exact, &to_plot_three_eight)/exact.len() as f64);


    let mut fg = Figure::new();
    let axis = fg.axes2d();
    plot_line_on(
        axis,
        &exact,
        &[
            Caption("Exact solution"),
            Color("black"),
        ],
    );
    plot_line_points_on(
        axis,
        &to_plot_classic,
        &[
            Caption(&format!(
                "Classic ERK with h={}", h
            )),
            Color("red"),
        ],
    );

    plot_line_points_on(
        axis,
        &to_plot_england,
        &[
            Caption(&format!(
                "England ERK with h={}", h
            )),
            Color("green"),
        ],
    );

    plot_line_points_on(
        axis,
        &to_plot_three_eight,
        &[
            Caption(&format!(
                "3/8 ERK with h={}", h
            )),
            Color("blue"),
        ],
    );

    let filename = IMAGE_DIR.to_owned().add(name).add(".png");
    fg.save_to_png(&filename, 1200, 800)
        .expect("Unable to save file");
}
