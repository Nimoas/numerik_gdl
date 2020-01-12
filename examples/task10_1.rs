use ngdl_rust::definitions::{Function, InitialValueSystemProblem, SampleableFunction};
use ngdl_rust::explicit_runge_kutta::make_classic_runge_kutta;
use num::abs_sub;
use num::pow;
use std::error::Error;

//const IMAGE_DIR: &str = "./img_task10_1/";

const T_TARGET: f64 = 0.3;
const H: f64 = 0.001;

const EXACT_X: f64 = 0.9886739393819;
const EXACT_Y: f64 = 0.00003447715743689;
const EXACT_Z: f64 = 0.01129158346063;

fn main() -> Result<(), Box<dyn Error>> {
    //create_dir_all(IMAGE_DIR)?;

    println!("Solving for t_end = {} with h = {}", T_TARGET, H);
    println!("\tExact x(t_end) = {}", EXACT_X);
    println!("\tExact y(t_end) = {}", EXACT_Y);
    println!("\tExact z(t_end) = {}", EXACT_Z);

    test_runge_kutta();

    Ok(())
}

fn test_runge_kutta() {
    let rk_method = make_classic_runge_kutta(create_problem(), H);
    let data = rk_method.value_at(T_TARGET);

    println!("Explicit Euler method:");
    println!("\tx'(t_end) = {}", data[0]);
    println!("\ty'(t_end) = {}", data[1]);
    println!("\tz'(t_end) = {}", data[2]);

    println!("\n\terror_x = {:e}", abs_sub(data[0], EXACT_X));
    println!("\terror_y = {:e}", abs_sub(data[1], EXACT_Y));
    println!("\terror_z = {:e}", abs_sub(data[2], EXACT_Z));
}

fn create_problem() -> InitialValueSystemProblem<Function<(f64, Vec<f64>)>> {
    let dfx: Function<(f64, Vec<f64>)> = |(_t, r)| -0.04 * r[0] + pow(10.0f64, 4) * r[1] * r[2];
    let dfy: Function<(f64, Vec<f64>)> =
        |(_t, r)| 0.04 * r[0] - pow(10.0f64, 4) * r[1] * r[2] - 3.0 * pow(10.0f64, 7) * r[1] * r[1];
    let dfz: Function<(f64, Vec<f64>)> = |(_t, r)| 3.0 * pow(10.0f64, 7) * r[1] * r[1];

    InitialValueSystemProblem::new(0.0, vec![1.0, 0.0, 0.0], vec![dfx, dfy, dfz])
}
