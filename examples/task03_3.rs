use gnuplot::AutoOption::Auto;
use gnuplot::PlotOption::{Caption, Color};
use gnuplot::TickOption::Format;
use gnuplot::{AxesCommon, Figure};
use ngdl_rust::definitions::{
    Function1D, Function2D, InitialValueProblem, Interval, Point2D, SimpleDifferentiableFunction,
};
use ngdl_rust::euler_explicit::explicit_euler_interval_test_run;
use ngdl_rust::implicit_euler::implicit_euler_interval_test_run;
use ngdl_rust::plot_util::{plot_line_on, plot_line_points_on, plot_points_on};
use ngdl_rust::util::sample_function;
use ngdl_rust::{cos, exp, sin};
use num::abs_sub;
use std::error::Error;
use std::f64::consts::E;
use std::fs::create_dir_all;
use std::ops::Add;

const IMAGE_DIR: &str = "./img_task03_3/";

fn main() -> Result<(), Box<dyn Error>> {
    create_dir_all(IMAGE_DIR)?;

    let t_target = 10.0;

    // y(x) = 1 / (1 - x) for x = 1/2
    let exact: Function1D = |x| sin!(x) + (1.0 / exp!(1000.0 * x));
    let exact_samples = 10000;
    let exact_sampled = sample_function(exact, Interval::new(0.0, 10.0), exact_samples);

    // t is ignored
    let ivp: InitialValueProblem<Function2D> =
        InitialValueProblem::new(0.0, 1.0, |(x, y)| -1000.0 * y + 1000.0 * sin!(x) + cos!(x));
    let ivp_implicit: InitialValueProblem<SimpleDifferentiableFunction<(f64, f64)>> =
        InitialValueProblem::new(
            0.0,
            1.0,
            SimpleDifferentiableFunction::new(
                |(t, x)| -1000.0 * x + 1000.0 * sin!(t) + cos!(t),
                |(_, _)| -1000.0,
            ),
        );

    let hs: Vec<f64> = vec![1.0, 0.1, 0.01, 0.001, 0.00199, 0.002, 0.0021];

    let approximations_explicit = explicit_euler_interval_test_run(ivp, &hs, t_target, 0);
    let approximations_implicit = implicit_euler_interval_test_run(ivp_implicit, &hs, t_target, 0);

    let explicit_errors: Vec<Vec<Point2D>> = approximations_explicit
        .iter()
        .map(|ps| {
            ps.iter()
                .map(|p| Point2D {
                    x: p.x,
                    y: abs_sub(p.y, exact(p.x)),
                })
                .collect()
        })
        .collect();
    let implicit_errors: Vec<Vec<Point2D>> = approximations_implicit
        .iter()
        .map(|ps| {
            ps.iter()
                .map(|p| Point2D {
                    x: p.x,
                    y: abs_sub(p.y, exact(p.x)),
                })
                .collect()
        })
        .collect();

    plot_helper("explicit", &exact_sampled, approximations_explicit, &hs);
    plot_helper("implicit", &exact_sampled, approximations_implicit, &hs);

    error_plot_helper("explicit_error", explicit_errors, &hs);
    error_plot_helper("implicit_error", implicit_errors, &hs);

    Ok(())
}

fn plot_helper(
    name: &str,
    exact_sampled: &[Point2D],
    approximations: Vec<Vec<Point2D>>,
    hs: &[f64],
) {
    for (data, h) in approximations.iter().zip(hs) {
        let mut fg = Figure::new();
        let axis = fg
            .axes2d()
            .set_y_ticks(Some((Auto, 0)), &[Format("%g")], &[]);
        plot_line_points_on(
            axis,
            data,
            &[
                Caption(&format!("Approximation with h={}", h)),
                Color("red"),
            ],
        );
        plot_line_on(
            axis,
            exact_sampled,
            &[Caption("Exact solution"), Color("green")],
        );

        let filename = IMAGE_DIR.to_owned().add(name).add(&format!("_{}.png", h));

        fg.save_to_png(&filename, 1200, 800)
            .expect("Unable to save file");
    }
}

fn error_plot_helper(name: &str, approximation_errors: Vec<Vec<Point2D>>, hs: &[f64]) {
    for (errors, h) in approximation_errors.iter().zip(hs) {
        let mut fg = Figure::new();
        let axis = fg
            .axes2d()
            .set_y_ticks(Some((Auto, 0)), &[Format("%g")], &[])
            .set_y_log(Some(10.0));
        plot_points_on(
            axis,
            errors,
            &[
                Caption(&format!("Approximation with h={}", h)),
                Color("red"),
            ],
        );
        let filename = IMAGE_DIR.to_owned().add(name).add(&format!("_{}.png", h));

        fg.save_to_png(&filename, 1200, 800)
            .expect("Unable to save file");
    }
}
