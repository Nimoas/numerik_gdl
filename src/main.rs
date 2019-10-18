use gnuplot::{Figure, AxesCommon};
use gnuplot::PlotOption::{Caption, Color};
use std::fs::create_dir_all;
use std::error::Error;
use std::ops::Add;
use crate::definitions::{Function1D, Interval};
use crate::quadrature::{trapezoid_formula, kepler_formula, newton_three_eight_formula, QuadratureFormula, quadrature_test_run, get_convergence_order};
use rayon::prelude::*;

const IMAGE_DIR: &str = "./img_task01/";

mod quadrature;
mod definitions;

fn main() -> Result<(), Box<dyn Error>> {
    create_dir_all(IMAGE_DIR)?;

    let interval = Interval::new(1.0, 10.0);
    let f: Function1D = |x| x.ln();
    let exact: f64 = 10.0 * (10.0f64).ln() - 9.0;

    evaluate_quadrature_method(trapezoid_formula, &interval, f, exact, "trapezoid.png");
    evaluate_quadrature_method(kepler_formula, &interval, f, exact, "kepler.png");
    evaluate_quadrature_method(newton_three_eight_formula, &interval, f, exact, "newton.png");

    Ok(())
}

fn evaluate_quadrature_method(method: QuadratureFormula, interval: &Interval, f: Function1D, exact: f64, name: &str) {
    let test_data = quadrature_test_run(method, f, exact, &interval, 200);
    // dbg!(&test_data);

    let ps: Vec<f64> = test_data.par_iter().zip(test_data.par_iter().skip(1)).map(|(a, b)| get_convergence_order(a, b)).collect();
    let p_sum: f64 = ps.iter().sum();
    let p_mean: f64 = p_sum / ps.len() as f64;
    dbg!(p_mean);

    let x: Vec<usize> = test_data.iter().map(|td| td.splits_n).collect();
    let y: Vec<f64> = test_data.iter().map(|td| td.abs_error).collect();

    let filename = String::from(IMAGE_DIR).add(name);

    let mut fg = Figure::new();
    fg.axes2d()
        .set_y_log(Some(10.0))
        .points(&x, &y, &[Caption("Abs Err over #Intervals"), Color("red")]);
    fg.save_to_png(&filename, 600, 400).expect("Unable to save file");
}
