use gnuplot::Coordinate::Graph;
use gnuplot::Figure;
use gnuplot::PlotOption::{Caption, Color, PointSymbol};
use ngdl_rust::definitions::{ClosureSampleableFunction, Function1D, InitialValueSystemProblem, Interval, Point2D, ODEMethod};
use ngdl_rust::euler_explicit::make_explicit_euler_method_system;
use ngdl_rust::explicit_runge_kutta::make_classic_runge_kutta;
use ngdl_rust::plot_util::{plot_line_on, plot_line_points_on};
use ngdl_rust::util::sample_function;
use ngdl_rust::{powi, sqrt};
use std::error::Error;
use std::f64::consts::PI;
use std::fs::create_dir_all;
use std::ops::Add;

const IMAGE_DIR: &str = "./img_task06_4/";

fn main() -> Result<(), Box<dyn Error>> {
    create_dir_all(IMAGE_DIR)?;

    let h = 0.1;
    let t_target = 1.0;
    let interval = Interval::new(0.0, t_target);

    let exact: Function1D = |t| (powi!(t, 4) / 144.0) + PI / 2.0;

    let exact_sampled = sample_function(exact, interval, 1000);

    let euler_method = make_explicit_euler_method_system(create_problem(PI / 2.0), h);
    let euler_sampled: Vec<Point2D> = euler_method
        .interval(t_target, 0)
        .iter()
        .map(|v| v[0])
        .collect();

    let rk_method = make_classic_runge_kutta(create_problem(PI / 2.0), h);
    let rk_sampled: Vec<Point2D> = rk_method
        .interval(t_target, 0)
        .iter()
        .map(|v| v[0])
        .collect();

    let mut fg = Figure::new();
    let axis = fg.axes2d().set_legend(Graph(0.3), Graph(1.0), &[], &[]);
    plot_line_points_on(
        axis,
        &euler_sampled,
        &[
            Caption("Euler approximation"),
            Color("red"),
            PointSymbol('o'),
        ],
    );
    plot_line_points_on(
        axis,
        &rk_sampled,
        &[
            Caption("ERK approximation"),
            Color("green"),
            PointSymbol('o'),
        ],
    );
    plot_line_on(
        axis,
        &exact_sampled,
        &[Caption("Exact solution"), Color("black")],
    );

    let filename = IMAGE_DIR.to_owned().add("plot.png");

    fg.save_to_png(&filename, 1200, 800)
        .expect("Unable to save file");

    Ok(())
}

fn create_problem(
    x_0: f64,
) -> InitialValueSystemProblem<ClosureSampleableFunction<(f64, Vec<f64>), f64>> {
    let dfx: ClosureSampleableFunction<(f64, Vec<f64>), f64> =
        ClosureSampleableFunction::new(x_0, |(_t, v), _| {
            // println!("t={}, z={}", t, v[1]);
            v[1]
        }); // x' = z
    let dfz: ClosureSampleableFunction<(f64, Vec<f64>), f64> =
        ClosureSampleableFunction::new(x_0, |(_t, v), x_start| {
            // println!("t={}, x={}, x_start={}, sqrt={}", t, v[0], x_start, sqrt!(v[0] - x_start));
            sqrt!(v[0] - x_start)
        });
    InitialValueSystemProblem::new(0.0, vec![x_0, 0.0], vec![dfx, dfz])
}
