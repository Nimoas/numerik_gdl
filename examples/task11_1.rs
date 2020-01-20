use gnuplot::AutoOption::Fix;
use gnuplot::PlotOption::{Caption, Color, PointSymbol};
use gnuplot::{AxesCommon, Figure};
use ngdl_rust::definitions::Interval;
use ngdl_rust::definitions::{ClosureSampleableFunction, Function, Point2D};
use ngdl_rust::fac;
use ngdl_rust::plot_util::plot_points_on;
use ngdl_rust::stability_area::sample_stability_area;
use ngdl_rust::{ONE, TWO};
use num::complex::Complex64;
use std::error::Error;
use std::fs::create_dir_all;
use std::ops::Add;

const NUM_SAMPLES: usize = 1500;
const IMAGE_DIR: &str = "./img_task11_1/";
const GAMMA: Complex64 = Complex64::new(
    0.788675134594812882254574390250978727823800875635063438009,
    0.0,
);
const GAMMA_NEG: Complex64 = Complex64::new(
    0.211324865405187117745425609749021272176199124364936561990,
    0.0,
);

const RE_INTERVAL: Interval = Interval::new(-4.0, 3.0);
const IM_INTERVAL: Interval = Interval::new(-3.5, 3.5);

fn main() -> Result<(), Box<dyn Error>> {
    create_dir_all(IMAGE_DIR)?;

    let ss = 1..=4;

    for s in ss {
        plot_stability_area_rk(s);
    }
    plot_stability_area_rk_b_1();
    plot_stability_area_implicit_euler();
    plot_stability_area_implicit_midpoint();
    plot_stability_area_rk_sdirk();
    plot_stability_area_rk_sdirk_neg();

    Ok(())
}

fn plot_stability_region(name: &str, caption: &str, stability_region: &Vec<Point2D>) {
    let mut fg = Figure::new();
    let axis = fg
        .axes2d()
        .set_x_range(Fix(-4.0), Fix(3.0))
        .set_x_grid(true)
        .set_y_range(Fix(-3.5), Fix(3.5))
        .set_y_grid(true)
        .set_aspect_ratio(Fix(1.0));

    plot_points_on(
        axis,
        &stability_region,
        &[Caption(caption), Color("green"), PointSymbol('o')],
    );

    let filename = IMAGE_DIR.to_owned().add(name).add(".png");
    fg.save_to_png(&filename, 1000, 1000)
        .expect("Unable to save file");
}

fn plot_stability_area_rk(ss: usize) {
    let r: ClosureSampleableFunction<Complex64, usize> =
        ClosureSampleableFunction::new(ss, |z, s| {
            (0..=s)
                .map(|k| z.powu(k as u32) / Complex64::new(fac!(k) as f64, 0.0))
                .fold(Complex64::new(0.0, 0.0), |sum, z| sum + z)
                .norm()
        });

    let stability_region = sample_stability_area(r, NUM_SAMPLES, RE_INTERVAL, IM_INTERVAL);

    plot_stability_region(
        &ss.to_string(),
        &format!("ERK with s = {}", ss),
        &stability_region,
    );
}

fn plot_stability_area_rk_b_1() {
    let r: Function<Complex64> = |z| {
        ((0..=3)
            .map(|k| z.powu(k as u32) / Complex64::new(fac!(k) as f64, 0.0))
            .fold(Complex64::new(0.0, 0.0), |sum, z| sum + z)
            + z.powu(4) / 12.0)
            .norm()
    };

    let stability_region = sample_stability_area(r, NUM_SAMPLES, RE_INTERVAL, IM_INTERVAL);

    plot_stability_region("b_1", "Task 11, 1, b) 1.", &stability_region);
}

fn plot_stability_area_rk_sdirk() {
    let r: Function<Complex64> = |z| {
        (ONE - z / (GAMMA * z - ONE) + z * (TWO * GAMMA - ONE) / (TWO * (GAMMA * z - ONE).powu(2)))
            .norm()
    };

    let stability_region = sample_stability_area(r, NUM_SAMPLES, RE_INTERVAL, IM_INTERVAL);

    plot_stability_region("sdirk", "SDIRK 3rd order", &stability_region);
}

fn plot_stability_area_rk_sdirk_neg() {
    let r: Function<Complex64> = |z| {
        (ONE - z / (GAMMA_NEG * z - ONE)
            + z * (TWO * GAMMA_NEG - ONE) / (TWO * (GAMMA_NEG * z - ONE).powu(2)))
        .norm()
    };

    let stability_region = sample_stability_area(r, NUM_SAMPLES, RE_INTERVAL, IM_INTERVAL);

    plot_stability_region(
        "sdirk_2",
        "SDIRK 3rd order (other Gamma)",
        &stability_region,
    );
}

fn plot_stability_area_implicit_euler() {
    let r: Function<Complex64> = |z| (ONE / (ONE - z)).norm();

    let stability_region = sample_stability_area(r, NUM_SAMPLES, RE_INTERVAL, IM_INTERVAL);

    plot_stability_region("implicit_euler", "Implicit euler", &stability_region);
}

fn plot_stability_area_implicit_midpoint() {
    let r: Function<Complex64> = |z| ((-z - TWO) / (z - TWO)).norm();

    let stability_region = sample_stability_area(r, NUM_SAMPLES, RE_INTERVAL, IM_INTERVAL);

    plot_stability_region("implicit_midpoint", "Implicit midpoint", &stability_region);
}
