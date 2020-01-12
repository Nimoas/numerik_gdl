use gnuplot::AutoOption::Auto;
use gnuplot::PlotOption::{Caption, Color};
use gnuplot::TickOption::Format;
use gnuplot::{AxesCommon, Figure};
use ngdl_rust::adams_bashforth::{make_adams_bashforth_2_method, make_adams_bashforth_3_method};
use ngdl_rust::definitions::{Function, InitialValueSystemProblem, Point2D, SampleableFunction};
use ngdl_rust::euler_explicit::make_explicit_euler_method_system;
use ngdl_rust::explicit_runge_kutta::{
    make_2nd_order_runge_kutta, make_classic_runge_kutta, make_heun_method,
};
use ngdl_rust::hack::{make_adams_moulton_hack_method, make_milne_simpson_hack_method};
use ngdl_rust::milne_simpson::make_milne_simpson_method;
use ngdl_rust::nystroem::make_nystroem_3_method;
use ngdl_rust::plot_util::plot_line_points_on;
use ngdl_rust::util::{get_all_convergence_orders, get_convergence_order};
use ngdl_rust::{exp, powi};
use std::error::Error;
use std::f64::consts::E;
use std::fs::create_dir_all;
use std::ops::Add;
use num::abs_sub;

const IMAGE_DIR: &str = "./img_task09_1/";

const T_TARGET: f64 = 1.0;

macro_rules! test_method {
    ($hs: expr, $exact_value: expr, $name: expr, $method: expr) => {
        let err_euler: Vec<f64> = $hs
            .iter()
            .map(|h| {
                $method(create_problem, *h, |ivp, h| {
                    make_explicit_euler_method_system(ivp, h)
                })
            })
            .map(|method| method.value_at(T_TARGET))
            .map(|approx| abs_sub(approx[0], $exact_value))
            .collect();
        let conv_euler: Vec<f64> = get_all_convergence_orders(&err_euler.clone(), $hs);
        println!(
            "Avrg. convergence order with Euler method: {}",
            get_convergence_order(&err_euler.clone(), $hs)
        );

        let err_2nd: Vec<f64> = $hs
            .iter()
            .map(|h| {
                $method(create_problem, *h, |ivp, h| {
                    make_2nd_order_runge_kutta(ivp, h)
                })
            })
            .map(|method| method.value_at(T_TARGET))
            .map(|approx| abs_sub(approx[0], $exact_value))
            .collect();
        let conv_2nd: Vec<f64> = get_all_convergence_orders(&err_2nd.clone(), $hs);
        println!(
            "Avrg. convergence order with 2nd order RK method: {}",
            get_convergence_order(&err_euler.clone(), $hs)
        );

        let err_heun: Vec<f64> = $hs
            .iter()
            .map(|h| $method(create_problem, *h, |ivp, h| make_heun_method(ivp, h)))
            .map(|method| method.value_at(T_TARGET))
            .map(|approx| abs_sub(approx[0], $exact_value))
            .collect();
        let conv_heun: Vec<f64> = get_all_convergence_orders(&err_heun.clone(), $hs);
        println!(
            "Avrg. convergence order with Heun method: {}",
            get_convergence_order(&err_euler.clone(), $hs)
        );

        let err_crk: Vec<f64> = $hs
            .iter()
            .map(|h| {
                $method(create_problem, *h, |ivp, h| {
                    make_classic_runge_kutta(ivp, h)
                })
            })
            .map(|method| method.value_at(T_TARGET))
            .map(|approx| abs_sub(approx[0], $exact_value))
            .collect();
        let conv_crk: Vec<f64> = get_all_convergence_orders(&err_crk.clone(), $hs);
        println!(
            "Avrg. convergence order with classical RK method: {}",
            get_convergence_order(&err_euler.clone(), $hs)
        );

        plot_err($hs, err_euler, err_2nd, err_heun, err_crk, $name);
        plot_convergence($hs, conv_euler, conv_2nd, conv_heun, conv_crk, $name);
    };
}

fn main() -> Result<(), Box<dyn Error>> {
    create_dir_all(IMAGE_DIR)?;

    let exact_fn = |t: f64| exp!(1.0 - powi!(t, 3) / 3.0);
    let exact_val: f64 = exact_fn(T_TARGET);
    println!("Exact value at {}: {}", T_TARGET, exact_val);

    // Start at 3, because 1/2 and 1/4 make no sense for h
    let hs: Vec<f64> = (3..20).map(|i| 1.0 / powi!(2.0f64, i)).collect();

    println!("\nTesting Adams Bashforth 2nd order");
    test_method!(
        &hs,
        exact_val,
        "adams_bashforth_2nd_order",
        make_adams_bashforth_2_method
    );
    println!("\nTesting Adams Bashforth 3rd order");
    test_method!(
        &hs,
        exact_val,
        "adams_bashforth_3rd_order",
        make_adams_bashforth_3_method
    );
    println!("\nTesting (explicitlyfied) Adam Moulton 4th order (3 steps)");
    test_method!(
        &hs,
        exact_val,
        "adams_moulton_hack",
        make_adams_moulton_hack_method
    );
    println!("\nTesting Nyström 3rd order");
    test_method!(&hs, exact_val, "nystroem", make_nystroem_3_method);
    println!("\nTesting Milne Simpson");
    test_method!(&hs, exact_val, "milne_simpson", make_milne_simpson_method);
    println!("\nTesting (explicitlyfied) Milne Simpson");
    test_method!(
        &hs,
        exact_val,
        "milne_simpson_hack",
        make_milne_simpson_hack_method
    );

    Ok(())
}

fn plot_err(
    hs: &[f64],
    euler: Vec<f64>,
    order2: Vec<f64>,
    heun: Vec<f64>,
    crk: Vec<f64>,
    name: &str,
) {
    let mut fg = Figure::new();
    let axis = fg
        .axes2d()
        .set_x_log(Some(10.0))
        .set_y_log(Some(10.0))
        .set_y_ticks(Some((Auto, 10)), &[Format("%.1e")], &[])
        .set_x_ticks(Some((Auto, 10)), &[Format("%.1e")], &[]);

    plot_line_points_on(
        axis,
        &Point2D::make_vec(hs.to_vec(), euler),
        &[Caption("Euler"), Color("red")],
    );
    plot_line_points_on(
        axis,
        &Point2D::make_vec(hs.to_vec(), order2),
        &[Caption("2nd Order"), Color("blue")],
    );
    plot_line_points_on(
        axis,
        &Point2D::make_vec(hs.to_vec(), heun),
        &[Caption("Heun"), Color("violet")],
    );
    plot_line_points_on(
        axis,
        &Point2D::make_vec(hs.to_vec(), crk),
        &[Caption("CRK"), Color("green")],
    );
    let filename = IMAGE_DIR.to_owned().add(name).add("_data.png");
    fg.save_to_png(&filename, 1200, 800)
        .expect("Unable to save file");
}

fn plot_convergence(
    hs: &[f64],
    euler: Vec<f64>,
    order2: Vec<f64>,
    heun: Vec<f64>,
    crk: Vec<f64>,
    name: &str,
) {
    let mut fg = Figure::new();
    let axis =
        fg.axes2d()
            .set_x_log(Some(10.0))
            .set_x_ticks(Some((Auto, 10)), &[Format("%.1e")], &[]);

    plot_line_points_on(
        axis,
        &Point2D::make_vec(hs.to_vec(), euler),
        &[Caption("Euler"), Color("red")],
    );
    plot_line_points_on(
        axis,
        &Point2D::make_vec(hs.to_vec(), order2),
        &[Caption("2nd Order"), Color("blue")],
    );
    plot_line_points_on(
        axis,
        &Point2D::make_vec(hs.to_vec(), heun),
        &[Caption("Heun"), Color("violet")],
    );
    plot_line_points_on(
        axis,
        &Point2D::make_vec(hs.to_vec(), crk),
        &[Caption("CRK"), Color("green")],
    );
    let filename = IMAGE_DIR.to_owned().add(name).add("_convergence.png");
    fg.save_to_png(&filename, 1200, 800)
        .expect("Unable to save file");
}

fn create_problem() -> InitialValueSystemProblem<Function<(f64, Vec<f64>)>> {
    let df: Function<(f64, Vec<f64>)> = |(t, v)| -t * t * v[0];

    InitialValueSystemProblem::new(0.0, vec![E], vec![df])
}
