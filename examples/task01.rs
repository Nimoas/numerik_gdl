use gnuplot::PlotOption::{Caption, Color};
use gnuplot::{AxesCommon, Figure};
use ngdl_rust::definitions::{Function1D, Interval};
use ngdl_rust::ln;
use ngdl_rust::quadrature::{
    get_convergence_order, quadrature_test_run, KeplerFormula, NewtonThreeEightFormula,
    QuadratureFormula, TrapezoidFormula,
};
use rayon::prelude::*;
use std::error::Error;
use std::fs::create_dir_all;
use std::ops::Add;

const IMAGE_DIR: &str = "./img_task01/";

fn main() -> Result<(), Box<dyn Error>> {
    create_dir_all(IMAGE_DIR)?;

    let interval = Interval::new(1.0, 10.0);
    let f: Function1D = |x| ln!(x);
    let exact: f64 = 10.0 * ln!(10.0f64) - 9.0;

    println!("Integrating ln(x) on the interval {}", interval);
    println!("Exact solution: {}\n", exact);

    evaluate_quadrature_method(&TrapezoidFormula, interval, f, exact, "trapezoid");
    evaluate_quadrature_method(&KeplerFormula, interval, f, exact, "kepler");
    evaluate_quadrature_method(&NewtonThreeEightFormula, interval, f, exact, "newton");

    Ok(())
}

fn evaluate_quadrature_method<QT: QuadratureFormula<Function1D>>(
    method: &QT,
    interval: Interval,
    f: Function1D,
    exact: f64,
    name: &str,
) {
    println!("Doing a testrun with method {}", name);

    let test_data = quadrature_test_run(method, f, exact, interval, 200);

    let ps: Vec<f64> = test_data
        .par_iter()
        .zip(test_data.par_iter().skip(1))
        .map(|(a, b)| get_convergence_order(a, b))
        .collect();
    let p_sum: f64 = ps.iter().sum();
    let p_mean: f64 = p_sum / ps.len() as f64;
    let last = test_data.iter().last().expect("Could not get last result");
    println!(
        "\tFinal approximation: {0} with absolute error: {1:e}",
        last.value, last.abs_error
    );
    println!("\tMean numerical convergence order: {}", p_mean);

    let x: Vec<usize> = test_data.iter().map(|td| td.splits_n).collect();
    let y: Vec<f64> = test_data.iter().map(|td| td.abs_error).collect();

    let filename = String::from(IMAGE_DIR).add(name).add(".png");

    let mut fg = Figure::new();
    fg.axes2d().set_y_log(Some(10.0)).points(
        &x,
        &y,
        &[Caption("Abs Err over #Intervals"), Color("red")],
    );
    fg.save_to_png(&filename, 600, 400)
        .expect("Unable to save file");

    println!("\tSaved plot to {}\n", filename);
}
