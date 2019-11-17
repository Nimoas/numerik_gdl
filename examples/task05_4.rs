use gnuplot::Figure;
use gnuplot::PlotOption::{Caption, Color};
use ngdl_rust::definitions::{ClosureSampleableFunction, InitialValueSystemProblem, Point2D};
use ngdl_rust::explicit_runge_kutta::{make_explicit_runge_kutta_with_tableau, Tableau};
use ngdl_rust::plot_util::plot_line_on;
use ngdl_rust::{cos, sin};
use rayon::prelude::*;
use std::error::Error;
use std::fs::create_dir_all;
use std::ops::Add;

const IMAGE_DIR: &str = "./img_task05_4/";
const SIGMA: f64 = 0.07274;
const GRAVITY: f64 = 0.00981;
const OTHER_CONSTANT: f64 = 0.0000009982;

fn main() -> Result<(), Box<dyn Error>> {
    create_dir_all(IMAGE_DIR)?;

    let ks = vec![0.2, 0.3, 0.4, 0.5];
    let h = 0.1;
    let curve_lengths = vec![32.0, 23.0, 23.0, 18.0];

    ks.par_iter()
        .zip(curve_lengths.par_iter())
        .for_each(|(k, s)| evaluate(*k, h, *s));

    Ok(())
}

/// v = vec![r, z, angle]
fn create_problem(
    ks: f64,
) -> InitialValueSystemProblem<ClosureSampleableFunction<(f64, Vec<f64>), f64>> {
    let dfr: ClosureSampleableFunction<(f64, Vec<f64>), f64> =
        ClosureSampleableFunction::new(ks, |(_, v), _| cos!(v[2]));
    let dfz: ClosureSampleableFunction<(f64, Vec<f64>), f64> =
        ClosureSampleableFunction::new(ks, |(_, v), _| sin!(v[2]));
    let dfangle: ClosureSampleableFunction<(f64, Vec<f64>), f64> =
        ClosureSampleableFunction::new(ks, |(_, v), k| {
            if v[0] > 0.0 {
                2.0 * k - (OTHER_CONSTANT * GRAVITY * v[1]) / SIGMA - sin!(v[2]) / v[0]
            } else {
                k - (OTHER_CONSTANT * GRAVITY * v[1]) / (2.0 * SIGMA)
            }
        });
    InitialValueSystemProblem::new(0.0, vec![0.0, -1.0, 0.0], vec![dfr, dfz, dfangle])
}

fn evaluate(ks: f64, h: f64, s: f64) {
    let problem = create_problem(ks);
    let tableau = Tableau::new(
        vec![0.0, 0.5, 1.0, 1.0],                                     // cs
        vec![1.0 / 6.0, 2.0 / 3.0, 0.0, 1.0 / 6.0],                   // bs
        vec![vec![], vec![0.5], vec![0.0, 1.0], vec![0.0, 0.0, 1.0]], //as
    );

    let rk_method = make_explicit_runge_kutta_with_tableau(problem, h, tableau);

    let approximation = rk_method.interval(s, 0);

    let rs: Vec<f64> = approximation.iter().map(|v| v[0].y).collect();
    let zs: Vec<f64> = approximation.iter().map(|v| v[1].y).collect();

    let to_plot: Vec<_> = rs
        .iter()
        .zip(zs)
        .map(|(r, z)| Point2D::new(*r, z))
        .collect();

    let mut fg = Figure::new();
    let axis = fg.axes2d();
    plot_line_on(
        axis,
        &to_plot,
        &[
            Caption(&format!(
                "RK: k_s = {} mm^-^1 with h = {} and s = {}",
                ks, h, s
            )),
            Color("black"),
        ],
    );

    let filename = IMAGE_DIR.to_owned().add(&format!("{}.png", ks));
    fg.save_to_png(&filename, 1200, 800)
        .expect("Unable to save file");
}
