use gnuplot::Coordinate::Graph;
use gnuplot::PlotOption::{Caption, Color};
use gnuplot::{AxesCommon, Figure};
use ngdl_rust::definitions::InitialValueProblem;
use ngdl_rust::euler_explicit::explicit_euler_test_run;
use std::error::Error;
use std::f64::consts::E;
use std::fs::create_dir_all;
use std::ops::Add;

const IMAGE_DIR: &str = "./img_task02_6/";

fn main() -> Result<(), Box<dyn Error>> {
    create_dir_all(IMAGE_DIR)?;

    let ks = 1..=15;

    // y(x) = 1 / (1 - x) for x = 1/2
    let exact = 2.0f64;
    // t is ignored
    let ivp = InitialValueProblem::new(0.0, 1.0, |_, x| x * x);

    // Only three samples is kinda boring, if we use a computer anyway...
    let hs: Vec<f64> = ks.clone().map(|n| 1.0 / ((2.0f64).powi(n))).collect();

    let approximations = explicit_euler_test_run(ivp, &hs, 0.5);

    let abs_errors: Vec<f64> = approximations.iter().map(|x| (exact - *x).abs()).collect();

    // This comes from the formula on task 6.
    // |f| = 2 and |f'| = 4 on the interval.
    let error_bounds: Vec<f64> = hs.iter().map(|h| 2.0 * (E.powi(2) - 1.0) * *h).collect();

    hs.iter()
        .zip(ks)
        .map(|(h, k)| format!("Absolute error for h=2^{}={}", k, h))
        .zip(&abs_errors)
        .for_each(|(str, e)| println!("{0} is {1:e}", str, e));

    let filename = String::from(IMAGE_DIR).add("error_plot.png");
    let mut fg = Figure::new();
    fg.axes2d()
        .set_y_log(Some(10.0))
        .set_x_log(Some(10.0))
        .set_legend(Graph(1.0), Graph(0.15), &[], &[])
        .lines_points(
            &hs,
            &abs_errors,
            &[Caption("Absolute error of the approximation"), Color("red")],
        )
        .lines_points(
            &hs,
            &error_bounds,
            &[
                Caption("Theoretical upper bound of the error"),
                Color("black"),
            ],
        );
    fg.save_to_png(&filename, 600, 400)
        .expect("Unable to save file");

    Ok(())
}
