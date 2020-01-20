use gnuplot::PlotOption::{Caption, Color};
use gnuplot::{AxesCommon, Figure};
use ngdl_rust::definitions::{Function, InitialValueSystemProblem, Point2D};
use ngdl_rust::embedded_rk::{make_dopri5, make_embedded_rk_1st_order};
use ngdl_rust::plot_util::plot_line_points_on;
use ngdl_rust::{powf, powi};
use std::error::Error;
use std::fs::create_dir_all;
use std::ops::Add;

const IMAGE_DIR: &str = "./img_task12_2/";
const MU: f64 = 0.012277471;
const H_START: f64 = 0.1;
const TOLERANCE: f64 = 0.001;
const T_TARGET: f64 = 17.06522;

fn main() -> Result<(), Box<dyn Error>> {
    create_dir_all(IMAGE_DIR)?;

    let mut dopri = make_dopri5(create_problem, H_START, TOLERANCE);
    let approximation_dop = dopri.interval(T_TARGET, 0);

    let xs_dop: Vec<f64> = approximation_dop.iter().map(|v| v[0].y).collect();
    let ys_dop: Vec<f64> = approximation_dop.iter().map(|v| v[2].y).collect();

    let to_plot_dop: Vec<_> = xs_dop
        .iter()
        .zip(ys_dop)
        .map(|(x, y)| Point2D::new(*x, y))
        .collect();

    let mut fg = Figure::new();
    let axis = fg.axes2d();
    plot_line_points_on(axis, &to_plot_dop, &[Caption("DOPRI5"), Color("green")]);

    let filename = IMAGE_DIR.to_owned().add("plot.png");
    fg.save_to_png(&filename, 1200, 800)
        .expect("Unable to save file");

    // hs
    let hs_dop: Vec<f64> = approximation_dop
        .iter()
        .skip(1)
        .map(|v| v[0].x)
        .zip(approximation_dop.iter().map(|v| v[0].x))
        .map(|(t2, t1)| t2 - t1)
        .collect();

    let hs_to_plot_dop: Vec<_> = hs_dop
        .iter()
        .enumerate()
        .map(|(x, y)| Point2D::new(x as f64, *y))
        .collect();

    let mut fg = Figure::new();
    let axis = fg.axes2d().set_y_log(Some(10.0));
    plot_line_points_on(axis, &hs_to_plot_dop, &[Caption("DOPRI5"), Color("green")]);

    let filename = IMAGE_DIR.to_owned().add("steps.png");
    fg.save_to_png(&filename, 1200, 800)
        .expect("Unable to save file");

    Ok(())
}

fn create_problem() -> InitialValueSystemProblem<Function<(f64, Vec<f64>)>> {
    let dfx1: Function<(f64, Vec<f64>)> = |(_t, v)| v[1];
    let dfy1: Function<(f64, Vec<f64>)> = |(_t, v)| {
        v[0] + 2.0 * v[3]
            - (1.0 - MU) * (v[0] + MU) / powf!(powi!(v[0] + MU, 2) + powi!(v[2], 2), 1.5)
            - MU * (v[0] - 1.0 + MU) / powf!(powi!(v[0] - 1.0 + MU, 2) + powi!(v[2], 2), 1.5)
    };

    let dfx2: Function<(f64, Vec<f64>)> = |(_t, v)| v[3];
    let dfy2: Function<(f64, Vec<f64>)> = |(_t, v)| {
        v[2] - 2.0 * v[2]
            - (1.0 - MU) * v[2] / powf!(powi!(v[0] + MU, 2) + powi!(v[2], 2), 1.5)
            - MU * v[2] / powf!(powi!(v[0] - 1.0 + MU, 2) + powi!(v[2], 2), 1.5)
    };

    InitialValueSystemProblem::new(
        0.0,
        vec![0.994, 0.0, 0.0, -2.001585106],
        vec![dfx1, dfy1, dfx2, dfy2],
    )
}
