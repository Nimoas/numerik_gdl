use gnuplot::Coordinate::Graph;
use gnuplot::Figure;
use gnuplot::PlotOption::{Caption, Color, PointSymbol};
use ngdl_rust::definitions::{Function2D, InitialValueProblem, Interval, Point2D};
use ngdl_rust::euler_explicit::explicit_euler_interval;
use ngdl_rust::plot_util::{plot_line_on, plot_line_points_on};
use ngdl_rust::util::sample_closure;
use ngdl_rust::{powf, sqrt};
use num::pow;
use std::error::Error;
use std::fs::create_dir_all;
use std::ops::Add;

const IMAGE_DIR: &str = "./img_task03_4/";

fn main() -> Result<(), Box<dyn Error>> {
    create_dir_all(IMAGE_DIR)?;

    // @Me: Wrong! See notes from the exercise.
    let exact = |x: f64, p: Point2D| 1.0 / pow(x + (1.0 / sqrt!(p.y)) - p.x, 2);
    let exact_samples = 1000;
    let exact_sampled = sample_closure(
        exact,
        Interval::new(0.0, 1.0),
        exact_samples,
        Point2D { x: 0.0, y: 1.0 },
    );

    // t is ignored
    let ivp: InitialValueProblem<Function2D> =
        InitialValueProblem::new(0.0, 1.0, |(_, x)| powf!(x, 1.5));
    let h = 0.2;

    let approximation = explicit_euler_interval(ivp, h, 1.0, 0);

    let mut fg = Figure::new();
    let axis = fg.axes2d().set_legend(Graph(0.3), Graph(1.0), &[], &[]);
    plot_line_points_on(
        axis,
        &approximation,
        &[Caption("Approximation"), Color("red"), PointSymbol('o')],
    );
    plot_line_on(
        axis,
        &exact_sampled,
        &[Caption("Exact solution"), Color("black")],
    );
    for p in approximation.iter().skip(1) {
        let sampled = sample_closure(exact, Interval::new(p.x, 1.0), exact_samples, *p);
        plot_line_on(axis, &sampled, &[Color("blue")]);
    }

    let filename = IMAGE_DIR.to_owned().add("plot.png");

    fg.save_to_png(&filename, 1200, 800)
        .expect("Unable to save file");

    Ok(())
}
