use ngdl_rust::definitions::{Function2D, InitialValueProblem};
use ngdl_rust::euler_explicit::explicit_euler_test_run;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // t is ignored
    let ivp: InitialValueProblem<Function2D> = InitialValueProblem::new(0.0, 1.0, |(_, x)| x * x);

    // Only three samples is kinda boring, if we use a computer anyway...
    let hs: Vec<f64> = (1..=15).map(|n| 1.0 / ((2.0f64).powi(n))).collect();

    dbg!(explicit_euler_test_run(ivp, &hs, 0.5));

    Ok(())
}
