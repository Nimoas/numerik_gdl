use gnuplot::PlotOption::{Caption, Color};
use gnuplot::{AxesCommon, Figure};
use ngdl_rust::definitions::{BoundaryValueProblem, Function1D, Interval};
use ngdl_rust::finite_differences_method::solve_bvp;
use ngdl_rust::util::make_supporting_points;
use std::error::Error;
use std::fs::create_dir_all;
use std::ops::Add;

const IMAGE_DIR: &str = "./img_task02_4/";

fn main() -> Result<(), Box<dyn Error>> {
    create_dir_all(IMAGE_DIR)?;

    // CONFIG PARAMETERS
    let exact_f: Function1D = |x| 4.0 * x.powf(2.5) / 15.0 - 19.0 / 15.0 * x + 1.0;
    let interval = Interval::new(0.0, 1.0);
    let bvp = BoundaryValueProblem::new(|x| x.sqrt(), interval, 1.0, 0.0);
    let n_grid = 99;

    // Calling the method to get a solution.
    let solution = solve_bvp(bvp, n_grid).expect("No solution found!");

    // Plotting
    let x = &make_supporting_points(n_grid + 1, interval)[1..n_grid + 1];
    let exact: Vec<f64> = x.iter().map(|x| exact_f(*x)).collect();
    let abs_error: Vec<f64> = exact
        .iter()
        .zip(&solution)
        .map(|(x, y)| (x - y).abs())
        .collect();

    let filename = String::from(IMAGE_DIR).add("plot.png");
    let mut fg = Figure::new();
    fg.axes2d()
        //.set_y_log(Some(10.0))
        .points(x, &solution, &[Caption("Approximation"), Color("red")])
        .lines(x, &exact, &[Caption("Exact"), Color("black")]);
    fg.save_to_png(&filename, 600, 400)
        .expect("Unable to save file");

    let filename2 = String::from(IMAGE_DIR).add("error.png");
    let mut fg2 = Figure::new();
    fg2.axes2d().set_y_log(Some(10.0)).points(
        x,
        &abs_error,
        &[Caption("Absolute error over x"), Color("red")],
    );
    fg2.save_to_png(&filename2, 600, 400)
        .expect("Unable to save file");

    Ok(())
}
