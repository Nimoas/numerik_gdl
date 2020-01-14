use gnuplot::AutoOption::Fix;
use gnuplot::Coordinate::Axis;
use gnuplot::PlotOption::{Caption, Color};
use gnuplot::{AxesCommon, Figure};
use ngdl_rust::definitions::Point2D;
use ngdl_rust::plot_util::plot_line_on;
use ngdl_rust::ZERO;
use ngdl_rust::{cos, sin};
use num::complex::Complex64;
use std::error::Error;
use std::f64::consts::PI;
use std::fs::create_dir_all;
use std::ops::Add;

const NUM_SAMPLES: usize = 1500;
const IMAGE_DIR: &str = "./img_task11_2/";

fn main() -> Result<(), Box<dyn Error>> {
    create_dir_all(IMAGE_DIR)?;

    // AB
    let (pts, p0, p1) = make_adams_wok(&vec![1.0, -1.0], &vec![0.0, 1.0]);
    plot_stability_region("adam_bashforth_1", "Adam-Bashforth k=1", &pts, p0, p1);

    let (pts, p0, p1) = make_adams_wok(
        &vec![1.0, -1.0, 0.0],
        &vec![23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0],
    );
    plot_stability_region("adam_bashforth_3", "Adam-Bashforth k=3", &pts, p0, p1);

    let (pts, p0, p1) = make_adams_wok(
        &vec![1.0, -1.0, 0.0, 0.0, 0.0],
        &vec![
            1901.0 / 720.0,
            -2774.0 / 720.0,
            2616.0 / 720.0,
            -1274.0 / 720.0,
            251.0 / 720.0,
        ],
    );
    plot_stability_region("adam_bashforth_5", "Adam-Bashforth k=5", &pts, p0, p1);

    // From http://www.mymathlib.com/c_source/diffeq/adams/adams_7_steps.c
    let (pts, p0, p1) = make_adams_wok(
        &vec![1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        &vec![
            198721.0 / 60480.0,
            -447288.0 / 60480.0,
            705549.0 / 60480.0,
            -688256.0 / 60480.0,
            407139.0 / 60480.0,
            -134472.0 / 60480.0,
            19087.0 / 60480.0,
        ],
    );
    plot_stability_region("adam_bashforth_7", "Adam-Bashforth k=7", &pts, p0, p1);

    // AM
    let (pts, p0, p1) = make_adams_wok(&vec![1.0, -1.0], &vec![1.0, 0.0]);
    plot_stability_region("adam_moulton_1", "Adam-Moulton k=1", &pts, p0, p1);

    let (pts, p0, p1) = make_adams_wok(
        &vec![1.0, -1.0, 0.0],
        &vec![5.0 / 12.0, 2.0 / 3.0, 1.0 / 12.0],
    );
    plot_stability_region("adam_moulton_3", "Adam-Moulton k=3", &pts, p0, p1);

    let (pts, p0, p1) = make_adams_wok(
        &vec![1.0, -1.0, 0.0, 0.0, 0.0],
        &vec![
            251.0 / 720.0,
            646.0 / 720.0,
            -264.0 / 720.0,
            106.0 / 720.0,
            -19.0 / 720.0,
        ],
    );
    plot_stability_region("adam_moulton_5", "Adam-Moulton k=5", &pts, p0, p1);

    let (pts, p0, p1) = make_adams_wok(
        &vec![1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        &vec![
            19087.0 / 60480.0,
            65112.0 / 60480.0,
            -46461.0 / 60480.0,
            37504.0 / 60480.0,
            -20211.0 / 60480.0,
            6312.0 / 60480.0,
            -863.0 / 60480.0,
        ],
    );
    plot_stability_region("adam_moulton_7", "Adam-Moulton k=7", &pts, p0, p1);

    Ok(())
}

fn plot_stability_region(
    name: &str,
    caption: &str,
    stability_region: &[Point2D],
    p0: Point2D,
    p1: Point2D,
) {
    let mut fg = Figure::new();
    let axis = fg
        .axes2d()
        .set_x_grid(true)
        .set_y_grid(true)
        .set_aspect_ratio(Fix(1.0));

    plot_line_on(axis, stability_region, &[Caption(caption), Color("green")]);
    axis.arrow(Axis(p0.x), Axis(p0.y), Axis(p1.x), Axis(p1.y), &[]);

    let filename = IMAGE_DIR.to_owned().add(name).add(".png");
    fg.save_to_png(&filename, 1000, 1000)
        .expect("Unable to save file");
}

fn make_adams_wok(alphas: &[f64], betas: &[f64]) -> (Vec<Point2D>, Point2D, Point2D) {
    let mut points = vec![];

    let roh = |phi: f64| {
        alphas
            .iter()
            .rev()
            .enumerate()
            .map(|(idx, alpha)| {
                Complex64::new(cos!(phi), sin!(phi))
                    .powu(idx as u32)
                    .scale(*alpha)
            })
            .fold(ZERO, |acc, val| acc + val)
    };

    let sigma = |phi: f64| {
        betas
            .iter()
            .rev()
            .enumerate()
            .map(|(idx, beta)| {
                Complex64::new(cos!(phi), sin!(phi))
                    .powu(idx as u32)
                    .scale(*beta)
            })
            .fold(ZERO, |acc, val| acc + val)
    };

    let wok = |phi| roh(phi) / sigma(phi);

    for k in 0..=NUM_SAMPLES {
        let sample = wok(2.0 * PI * (k as f64 / NUM_SAMPLES as f64));
        points.push(Point2D::new(sample.re, sample.im));
    }

    let p0 = wok(0.0);
    let p1 = wok(PI / 4.0);

    (
        points,
        Point2D::new(p0.re, p0.im),
        Point2D::new(p1.re, p1.im),
    )
}
