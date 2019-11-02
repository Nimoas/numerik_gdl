use gnuplot::AutoOption::Auto;
use gnuplot::PlotOption::{Caption, Color, LineWidth};
use gnuplot::TickOption::Format;
use gnuplot::{AxesCommon, Figure};
use ngdl_rust::definitions::{
    Function1D, Function2D, InitialValueProblem, Point2D, SimpleDifferentiableFunction2D,
};
use ngdl_rust::euler_explicit::explicit_euler_interval_test_run;
use ngdl_rust::implicit_euler::implicit_euler_interval_test_run;
use ngdl_rust::{cos, exp, sin};
use std::error::Error;
use std::f64::consts::E;
use std::fs::create_dir_all;

const IMAGE_DIR: &str = "./img_task03_3/";

fn main() -> Result<(), Box<dyn Error>> {
    create_dir_all(IMAGE_DIR)?;

    let t_target = 10.0;

    // y(x) = 1 / (1 - x) for x = 1/2
    let exact_sample_rate = 0.001;
    let exact: Function1D = |x| sin!(x) + (1.0 / exp!(1000.0 * x));
    let mut exact_ts = Vec::new();
    let mut excat_vals = Vec::new();
    let mut t = 0.0;
    while t <= t_target {
        exact_ts.push(t);
        excat_vals.push(exact(t));
        t += exact_sample_rate;
    }

    // t is ignored
    let ivp: InitialValueProblem<Function2D> =
        InitialValueProblem::new(0.0, 1.0, |x, y| -1000.0 * y + 1000.0 * sin!(x) + cos!(x));
    let ivp_implicit: InitialValueProblem<SimpleDifferentiableFunction2D> =
        InitialValueProblem::new(
            0.0,
            1.0,
            SimpleDifferentiableFunction2D::new(
                |t, x| -1000.0 * x + 1000.0 * sin!(t) + cos!(t),
                |_, _| -1000.0,
            ),
        );

    let hs: Vec<f64> = vec![1.0, 0.1, 0.01, 0.001, 0.00199, 0.002, 0.0021];

    let approximations_explicit = explicit_euler_interval_test_run(ivp, &hs, t_target, 0);
    let approximations_implicit = implicit_euler_interval_test_run(ivp_implicit, &hs, t_target, 0);

    plot_helper(&mut exact_ts, &mut excat_vals, approximations_explicit, &hs);
    plot_helper(&mut exact_ts, &mut excat_vals, approximations_implicit, &hs);

    Ok(())
}

fn plot_helper(
    exact_ts: &[f64],
    excat_vals: &[f64],
    approximations: Vec<Vec<Point2D>>,
    hs: &[f64],
) {
    let mut fg = Figure::new();
    let axis = fg
        .axes2d()
        .set_y_ticks(Some((Auto, 0)), &[Format("%g")], &[]);
    for (data, h) in approximations.iter().zip(hs) {
        axis.lines_points(
            data.iter().map(|d| d.x).collect::<Vec<f64>>(),
            data.iter().map(|d| d.y).collect::<Vec<f64>>(),
            &[
                Caption(&format!("Approximation with h={}", h)),
                Color("red"),
            ],
        );
    }
    axis.lines(
        exact_ts,
        excat_vals,
        &[Caption("Exact solution"), Color("green"), LineWidth(2.0)],
    );
    fg.show().expect("Unable to save file");
}
