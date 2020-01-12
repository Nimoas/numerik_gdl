use gnuplot::Coordinate::Graph;
use gnuplot::Figure;
use gnuplot::PlotOption::{Caption, Color, PointSymbol};
use ngdl_rust::definitions::{
    Function1D, Function2D, InitialValueProblem, Interval, SimpleDifferentiableFunction,
    SimpleDifferentiableFunction2D,
};
use ngdl_rust::euler_explicit::{explicit_euler, explicit_euler_interval};
use ngdl_rust::exp;
use ngdl_rust::modified_explicit_euler::{
    modified_explicit_euler, modified_explicit_euler_interval,
};
use ngdl_rust::plot_util::{plot_line_on, plot_line_points_on};
use ngdl_rust::util::{get_convergence_order, sample_function};
use num::abs_sub;
use std::error::Error;
use std::f64::consts::E;
use std::fs::create_dir_all;
use std::ops::Add;

const IMAGE_DIR: &str = "./img_task05_1/";

fn main() -> Result<(), Box<dyn Error>> {
    create_dir_all(IMAGE_DIR)?;

    let exact: Function1D = |x: f64| 0.25 * (-2.0 * x + 5.0 * exp!(2.0 * x - 2.0) - 1.0);
    let interval = Interval::new(1.0, 2.0);

    let exact_samples = 1000;
    let exact_sampled = sample_function(exact, interval, exact_samples);

    let h = 0.1;
    let t_target = 2.0;
    let dy: Function2D = |(x, y)| x + 2.0 * y;
    let ddy: Function2D = |(x, y)| 1.0 + 2.0 * x + 4.0 * y;

    let ivp: InitialValueProblem<Function2D> = InitialValueProblem::new(1.0, 0.5, dy);
    let approximation = explicit_euler_interval(ivp, h, t_target, 0);

    let ivp2: InitialValueProblem<SimpleDifferentiableFunction2D> =
        InitialValueProblem::new(1.0, 0.5, SimpleDifferentiableFunction::new(dy, ddy));
    let approximation2 = modified_explicit_euler_interval(ivp2, h, t_target, 0);

    let mut fg = Figure::new();
    let axis = fg.axes2d().set_legend(Graph(0.5), Graph(1.0), &[], &[]);
    plot_line_points_on(
        axis,
        &approximation,
        &[
            Caption("Approximation using the explicit euler method"),
            Color("red"),
            PointSymbol('o'),
        ],
    );
    plot_line_points_on(
        axis,
        &approximation2,
        &[
            Caption("Approximation using the improved explicit euler method"),
            Color("blue"),
            PointSymbol('x'),
        ],
    );
    plot_line_on(
        axis,
        &exact_sampled,
        &[Caption("Exact solution"), Color("black")],
    );

    let filename = IMAGE_DIR.to_owned().add("plot.png");

    fg.save_to_png(&filename, 1200, 800)
        .expect("Unable to save file");

    // Now convergence orders
    let hs: Vec<f64> = (1..500).map(|n| interval.span() / n as f64).collect();
    let exact_at_2: f64 = exact(t_target);
    let normal_euler_results: Vec<f64> = hs
        .iter()
        .map(|i| explicit_euler(ivp, *i, t_target))
        .collect();
    let modified_euler_results: Vec<f64> = hs
        .iter()
        .map(|i| modified_explicit_euler(ivp2, *i, t_target))
        .collect();

    let normal_abs_error: Vec<f64> = normal_euler_results
        .iter()
        .map(|aprox| abs_sub(*aprox, exact_at_2))
        .collect();
    let modified_abs_error: Vec<f64> = modified_euler_results
        .iter()
        .map(|aprox| abs_sub(*aprox, exact_at_2))
        .collect();

    println!(
        "Convergence order for the normal euler method: {}",
        get_convergence_order(&normal_abs_error, &hs)
    );
    println!(
        "Convergence order for the improved euler method: {}",
        get_convergence_order(&modified_abs_error, &hs)
    );

    Ok(())
}
