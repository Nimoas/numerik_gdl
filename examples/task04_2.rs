use ngdl_rust::definitions::{
    Function, Function1D, FunctionND, InitialValueSystemProblem, Interval, MultSampleableFunction,
    Subtractible,
};
use ngdl_rust::euler_explicit::{explicit_euler_system, get_residual_function};
use ngdl_rust::exp;
use ngdl_rust::quadrature::{quadrature, NewtonThreeEightFormula};
use ngdl_rust::util::euclidean_norm;
use std::error::Error;
use std::f64::consts::E;

fn main() -> Result<(), Box<dyn Error>> {
    let h = 0.1;
    let t_target = 1.0;

    let problem = create_problem();
    let exact: FunctionND<f64, Vec<f64>> =
        |t| vec![0.5 * (exp!(-2.0 * t) + 1.0), 0.5 * (-exp!(-2.0 * t) + 1.0)];

    let solution = explicit_euler_system(&problem, h, t_target);

    println!("Problem from task 4, subtask 2\n");
    println!("Step size: h = {}", h);
    println!("target t: t = {}\n", t_target);
    println!("Approximated solution: x_t = {}, y_t = {}", &solution[0], &solution[1]);

    let exact_value = exact(1.0);
    println!("Exact solution: x_t = {}, y_t = {}\n", &exact_value[0], &exact_value[1]);

    let num_error = solution.sub(exact_value);
    println!("Numerical error (in euclidean norm): {}", euclidean_norm(num_error));

    evaluate_first_error_bound(h);

    evaluate_second_error_bound(h);

    Ok(())
}

fn evaluate_second_error_bound(h: f64) {
    let interval = Interval::new(0.0, 1.0);
    let error_bound_2 = quadrature(
        &NewtonThreeEightFormula,
        &get_residual_function(create_problem, h),
        interval,
        1000,
    );

    println!("Second error bound: {}", error_bound_2);
}

fn evaluate_first_error_bound(h: f64) {
    let f_exp: Function1D = |x| exp!((1.0 - x) * 2.0f64);
    let to_integrate = MultSampleableFunction::new(f_exp, get_residual_function(create_problem, h));

    let interval = Interval::new(0.0, 1.0);
    let error_bound = quadrature(&NewtonThreeEightFormula, &to_integrate, interval, 1000);

    println!("First error bound: {}", error_bound);
}

fn create_problem() -> InitialValueSystemProblem<Function<(f64, Vec<f64>)>> {
    let dfx: Function<(f64, Vec<f64>)> = |(_, v)| -v[0] + v[1];
    let dfy: Function<(f64, Vec<f64>)> = |(_, v)| v[0] - v[1];
    InitialValueSystemProblem::new(0.0, vec![1.0, 0.0], vec![dfx, dfy])
}
