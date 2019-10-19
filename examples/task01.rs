use gnuplot::{Figure, AxesCommon};
use gnuplot::PlotOption::{Caption, Color};
use std::fs::create_dir_all;
use std::error::Error;
use std::ops::Add;
use rayon::prelude::*;
use ngdl_rust::definitions::{Interval, Function1D};
use ngdl_rust::quadrature::{trapezoid_formula, kepler_formula, newton_three_eight_formula, QuadratureFormula, quadrature_test_run, get_convergence_order};

const IMAGE_DIR: &str = "./img_task01/";

fn main() -> Result<(), Box<dyn Error>> {
    create_dir_all(IMAGE_DIR)?;

    let interval = Interval::new(1.0, 10.0);
    let f: Function1D = |x| x.ln();
    let exact: f64 = 10.0 * (10.0f64).ln() - 9.0;

    println!("Integrating ln(x) on the interval {}", interval);
    println!("Exact solution: {}\n", exact);

    evaluate_quadrature_method(trapezoid_formula, &interval, f, exact, "trapezoid");
    evaluate_quadrature_method(kepler_formula, &interval, f, exact, "kepler");
    evaluate_quadrature_method(newton_three_eight_formula, &interval, f, exact, "newton");

    Ok(())
}

fn evaluate_quadrature_method(method: QuadratureFormula, interval: &Interval, f: Function1D, exact: f64, name: &str) {
    println!("Doing a testrun with method {}", name);

    let test_data = quadrature_test_run(method, f, exact, &interval, 200);

    let ps: Vec<f64> = test_data.par_iter().zip(test_data.par_iter().skip(1)).map(|(a, b)| get_convergence_order(a, b)).collect();
    let p_sum: f64 = ps.iter().sum();
    let p_mean: f64 = p_sum / ps.len() as f64;
    let last = test_data.iter().last().expect("Could not get last result");
    println!("\tFinal approximation: {0} with absolute error: {1:e}", last.value, last.abs_error);
    println!("\tMean numerical convergence order: {}", p_mean);

    let x: Vec<usize> = test_data.iter().map(|td| td.splits_n).collect();
    let y: Vec<f64> = test_data.iter().map(|td| td.abs_error).collect();

    let filename = String::from(IMAGE_DIR).add(name).add(".png");

    let mut fg = Figure::new();
    fg.axes2d()
        .set_y_log(Some(10.0))
        .points(&x, &y, &[Caption("Abs Err over #Intervals"), Color("red")]);
    fg.save_to_png(&filename, 600, 400).expect("Unable to save file");

    println!("\tSaved plot to {}\n", filename);
}
