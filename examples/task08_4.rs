use gnuplot::Figure;
use gnuplot::PlotOption::{Caption, Color};
use ngdl_rust::adams_bashforth::make_adams_bashforth_2_method;
use ngdl_rust::definitions::{Function, InitialValueSystemProblem, ODEMethod, Point2D};
use ngdl_rust::euler_explicit::make_explicit_euler_method_system;
use ngdl_rust::explicit_runge_kutta::make_classic_runge_kutta;
use ngdl_rust::plot_util::{plot_line_on, plot_line_points_on};
use ngdl_rust::{powi, sqrt};
use std::error::Error;
use std::fs::create_dir_all;
use std::ops::Add;

const IMAGE_DIR: &str = "./img_task08_4/";

const GRAVITY: f64 = 0.00000000006672;
const MASS_EARTH: f64 = 5_980_000_000_000_000_000_000_000.0;
const MASS_MARS: f64 = 642_000_000_000_000_000_000_000.0;
const MASS_SUN: f64 = 1_990_000_000_000_000_000_000_000_000_000.0;

fn main() -> Result<(), Box<dyn Error>> {
    create_dir_all(IMAGE_DIR)?;

    let t_target = 3.0 * 59_400_000.0; // ~ Seconds in a year on Mars

    test_euler(t_target);
    test_runge_kutta(t_target);
    test_adams_bashforth(t_target);

    Ok(())
}

fn plot_data(data: Vec<Vec<Point2D>>, name: &str) {
    let x_e: Vec<f64> = data.iter().map(|v| v[0].y).collect();
    let y_e: Vec<f64> = data.iter().map(|v| v[1].y).collect();
    let earth = Point2D::make_vec(x_e, y_e);

    let x_m: Vec<f64> = data.iter().map(|v| v[2].y).collect();
    let y_m: Vec<f64> = data.iter().map(|v| v[3].y).collect();
    let mars = Point2D::make_vec(x_m, y_m);

    let x_s: Vec<f64> = data.iter().map(|v| v[4].y).collect();
    let y_s: Vec<f64> = data.iter().map(|v| v[5].y).collect();
    let sun = Point2D::make_vec(x_s, y_s);

    let mut fg = Figure::new();
    let axis = fg.axes2d();
    plot_line_on(axis, &earth, &[Caption("Earth"), Color("green")]);
    plot_line_on(axis, &mars, &[Caption("Mars"), Color("red")]);
    plot_line_points_on(axis, &sun, &[Caption("Sun"), Color("yellow")]);
    let filename = IMAGE_DIR.to_owned().add(name).add(".png");
    fg.save_to_png(&filename, 1200, 800)
        .expect("Unable to save file");
}

fn test_runge_kutta(t_target: f64) {
    let h = 10000.0;

    let rk_method = make_classic_runge_kutta(create_problem(), h);
    let data = rk_method.interval(t_target, 0);

    plot_data(data, "rk");
}

fn test_euler(t_target: f64) {
    let h = 100.0;

    let euler_method = make_explicit_euler_method_system(create_problem(), h);
    let data = euler_method.interval(t_target, 0);

    plot_data(data, "euler");
}

fn test_adams_bashforth(t_target: f64) {
    let h = 10000.0;

    let ab_method = make_adams_bashforth_2_method(create_problem, h, |ivp, h| {
        make_explicit_euler_method_system(ivp, h)
    });
    let data = ab_method.interval(t_target, 0);

    plot_data(data, "adams_bashforth");
}

/// v = vec![x_E, y_E, x_M, y_M, x_S, y_S, same for dt]
/// x_E = 0
/// y_E = 1
/// x_M = 2
/// y_M = 3
/// x_S = 4
/// y_S = 5
/// x'_E = 6
/// y'_E = 7
/// x'_M = 8
/// y'_M = 9
/// x'_S = 10
/// y'_S = 11
fn create_problem() -> InitialValueSystemProblem<Function<(f64, Vec<f64>)>> {
    let dfx_e: Function<(f64, Vec<f64>)> = |(_t, r)| r[6];
    let dfy_e: Function<(f64, Vec<f64>)> = |(_t, r)| r[7];
    let dfx_m: Function<(f64, Vec<f64>)> = |(_t, r)| r[8];
    let dfy_m: Function<(f64, Vec<f64>)> = |(_t, r)| r[9];
    let dfx_s: Function<(f64, Vec<f64>)> = |(_t, r)| r[10];
    let dfy_s: Function<(f64, Vec<f64>)> = |(_t, r)| r[11];

    let ddfx_e: Function<(f64, Vec<f64>)> = |(_t, r)| {
        (-1.0 * GRAVITY * MASS_EARTH * MASS_MARS * (r[0] - r[2])
            / powi!(sqrt!(powi!(r[0] - r[2], 2) + powi!(r[1] - r[3], 2)), 3)
            + GRAVITY * MASS_SUN * MASS_EARTH * (r[4] - r[0])
                / powi!(sqrt!(powi!(r[0] - r[4], 2) + powi!(r[1] - r[5], 2)), 3))
            / MASS_EARTH
    };
    let ddfy_e: Function<(f64, Vec<f64>)> = |(_t, r)| {
        (-1.0 * GRAVITY * MASS_EARTH * MASS_MARS * (r[1] - r[3])
            / powi!(sqrt!(powi!(r[0] - r[2], 2) + powi!(r[1] - r[3], 2)), 3)
            + GRAVITY * MASS_SUN * MASS_EARTH * (r[5] - r[1])
                / powi!(sqrt!(powi!(r[0] - r[4], 2) + powi!(r[1] - r[5], 2)), 3))
            / MASS_EARTH
    };
    let ddfx_m: Function<(f64, Vec<f64>)> = |(_t, r)| {
        (GRAVITY * MASS_EARTH * MASS_MARS * (r[0] - r[2])
            / powi!(sqrt!(powi!(r[0] - r[2], 2) + powi!(r[1] - r[3], 2)), 3)
            + GRAVITY * MASS_SUN * MASS_MARS * (r[4] - r[2])
                / powi!(sqrt!(powi!(r[4] - r[2], 2) + powi!(r[5] - r[3], 2)), 3))
            / MASS_MARS
    };
    let ddfy_m: Function<(f64, Vec<f64>)> = |(_t, r)| {
        (GRAVITY * MASS_EARTH * MASS_MARS * (r[1] - r[3])
            / powi!(sqrt!(powi!(r[0] - r[2], 2) + powi!(r[1] - r[3], 2)), 3)
            + GRAVITY * MASS_SUN * MASS_MARS * (r[5] - r[3])
                / powi!(sqrt!(powi!(r[4] - r[2], 2) + powi!(r[5] - r[3], 2)), 3))
            / MASS_MARS
    };
    let ddfx_s: Function<(f64, Vec<f64>)> = |(_t, r)| {
        (-1.0 * GRAVITY * MASS_SUN * MASS_EARTH * (r[4] - r[0])
            / powi!(sqrt!(powi!(r[4] - r[0], 2) + powi!(r[5] - r[1], 2)), 3)
            + -1.0 * GRAVITY * MASS_SUN * MASS_MARS * (r[4] - r[2])
                / powi!(sqrt!(powi!(r[4] - r[2], 2) + powi!(r[5] - r[3], 2)), 3))
            / MASS_SUN
    };
    let ddfy_s: Function<(f64, Vec<f64>)> = |(_t, r)| {
        (-1.0 * GRAVITY * MASS_SUN * MASS_EARTH * (r[5] - r[1])
            / powi!(sqrt!(powi!(r[4] - r[0], 2) + powi!(r[5] - r[1], 2)), 3)
            + -1.0 * GRAVITY * MASS_SUN * MASS_MARS * (r[5] - r[3])
                / powi!(sqrt!(powi!(r[4] - r[2], 2) + powi!(r[5] - r[3], 2)), 3))
            / MASS_SUN
    };

    InitialValueSystemProblem::new(
        0.0,
        vec![
            150.0 * powi!(10.0f64, 9),
            0.0,
            228.0 * powi!(10.0f64, 9),
            0.0,
            0.0,
            0.0,
            0.0, // from here ddf
            29.0 * powi!(10.0f64, 3),
            0.0,
            24.0 * powi!(10.0f64, 3),
            0.0,
            0.0,
        ],
        vec![
            dfx_e, dfy_e, dfx_m, dfy_m, dfx_s, dfy_s, ddfx_e, ddfy_e, ddfx_m, ddfy_m, ddfx_s,
            ddfy_s,
        ],
    )
}
