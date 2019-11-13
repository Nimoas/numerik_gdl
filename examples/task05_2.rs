use std::fs::create_dir_all;
use std::error::Error;
use ngdl_rust::definitions::{InitialValueSystemProblem, ClosureSampleableFunction, Point2D};
use ngdl_rust::{cos, sin};
use ngdl_rust::euler_explicit::explicit_euler_system_interval;
use gnuplot::Figure;
use ngdl_rust::plot_util::plot_line_on;
use gnuplot::PlotOption::{Caption, Color};
use rayon::prelude::*;

const IMAGE_DIR: &str = "./img_task05_1/";
const SIGMA: f64 = 0.07274;
const GRAVITY: f64 = 0.00981;
const OTHER_CONSTANT: f64 = 0.0000009982;

fn main() -> Result<(), Box<dyn Error>> {
    create_dir_all(IMAGE_DIR)?;

    let ks = vec![0.2, 0.3/*, 0.4, 0.5*/];
    let h = 0.0001;
    let s = 40.0;

    let hs  = vec![0.0001, 0.000105, 0.00001];

    hs.par_iter().for_each(|hn| evaluate(ks[0], *hn, s));

    Ok(())
}

/// v = vec![r, z, angle]
fn create_problem(ks: f64) -> InitialValueSystemProblem<ClosureSampleableFunction<(f64, Vec<f64>), f64>> {
    let dfr: ClosureSampleableFunction<(f64, Vec<f64>), f64> = ClosureSampleableFunction::new(ks, |(_, v), _| cos!(v[2]));
    let dfz: ClosureSampleableFunction<(f64, Vec<f64>), f64> = ClosureSampleableFunction::new(ks, |(_, v), _| sin!(v[2]));
    let dfangle: ClosureSampleableFunction<(f64, Vec<f64>), f64> = ClosureSampleableFunction::new(ks, |(_, v), k| if v[0] > 0.0 {
        2.0 * k - (OTHER_CONSTANT * GRAVITY * v[1]) / SIGMA - sin!(v[2]) / v[0]
    } else {
        k - (OTHER_CONSTANT * GRAVITY * v[1]) / (2.0 * SIGMA)
    });
    InitialValueSystemProblem::new(0.0, vec![0.0, -1.0, 0.0], vec![dfr, dfz, dfangle])
}

fn evaluate(ks: f64, h: f64, s: f64) {
    let problem = create_problem(ks);

    let approximation = explicit_euler_system_interval(&problem, h, s, 0);

    let rs: Vec<f64> = approximation.iter().map(|v| v[0].y).collect();
    let zs: Vec<f64> = approximation.iter().map(|v| v[1].y).collect();

    let to_plot: Vec<_> = rs.iter().zip(zs).map(|(r, z)| Point2D::new(*r, z)).collect();

    let mut fg = Figure::new();
    let axis = fg.axes2d();
    plot_line_on(
        axis,
        &to_plot,
        &[Caption(&format!("k_s = {} mm^-^1 with h = {}", ks, h)), Color("black")],
    );

    fg.show()
        .expect("Unable to save file");
}