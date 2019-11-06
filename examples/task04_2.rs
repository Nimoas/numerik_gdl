use ngdl_rust::definitions::{Function, InitialValueSystemProblem};
use ngdl_rust::euler_explicit::explicit_euler_system;
use std::error::Error;
use std::fs::create_dir_all;

const IMAGE_DIR: &str = "./img_task04_2/";

fn main() -> Result<(), Box<dyn Error>> {
    create_dir_all(IMAGE_DIR)?;

    let h = 0.1;
    let t_target = 1.0;
    let dfx: Function<(f64, Vec<f64>)> = |(_, v)| -v[0] + v[1];
    let dfy: Function<(f64, Vec<f64>)> = |(_, v)| v[0] - v[1];

    let problem = InitialValueSystemProblem::new(0.0, vec![1.0, 0.0], vec![dfx, dfy]);

    let solution = explicit_euler_system(problem, h, t_target);

    dbg!(solution);

    Ok(())
}
