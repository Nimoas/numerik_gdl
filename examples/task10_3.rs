use gnuplot::Figure;
use gnuplot::PlotOption::{Caption, Color};
use ngdl_rust::definitions::{Function, InitialValueSystemProblem, Interval, ODEMethod, Point2D};
use ngdl_rust::explicit_runge_kutta::make_classic_runge_kutta;
use ngdl_rust::hack::make_task_10_3_method;
use ngdl_rust::plot_util::{plot_line_on, plot_line_points_on};
use ngdl_rust::util::sample_function;
use ngdl_rust::{cos, exp, sin};
use std::error::Error;
use std::f64::consts::E;
use std::fs::create_dir_all;
use std::ops::Add;

const IMAGE_DIR: &str = "./img_task10_3/";

const T_TARGET: f64 = 5.0;
const H: f64 = 0.1;

fn main() -> Result<(), Box<dyn Error>> {
    create_dir_all(IMAGE_DIR)?;

    let alphas: Vec<f64> = vec![-1.0, -0.99, -0.9, 0.0, 0.9, 0.99];

    for a in alphas {
        test_for_a(a);
    }

    Ok(())
}

fn create_problem() -> InitialValueSystemProblem<Function<(f64, Vec<f64>)>> {
    let df: Function<(f64, Vec<f64>)> = |(t, v)| -v[0] + exp!(-t) * cos!(t);

    InitialValueSystemProblem::new(0.0, vec![0.0], vec![df])
}

fn test_for_a(a: f64) {
    let method = make_task_10_3_method(
        create_problem,
        H,
        |ivp, h| make_classic_runge_kutta(ivp, h),
        a,
    );

    let interval = method.interval(T_TARGET, 0);

    let points: Vec<Point2D> = interval.iter().map(|v| v[0]).collect();

    let exact_fn = |t: f64| exp!(-t) * sin!(t);
    let exact_points = sample_function(exact_fn, Interval::new(0.0, T_TARGET), 1000);

    let mut fg = Figure::new();
    let axis = fg.axes2d();
    plot_line_points_on(
        axis,
        &points,
        &[Caption(&format!("Approx. with a = {}", a)), Color("red")],
    );
    plot_line_on(axis, &exact_points, &[Caption("Exact"), Color("black")]);

    let filename = IMAGE_DIR.to_owned().add(&format!("{}", a)).add(".png");
    fg.save_to_png(&filename, 1200, 800)
        .expect("Unable to save file");
}
