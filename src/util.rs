use crate::definitions::{Closure1D, Function1D, Interval, Point2D};
use crate::{ln, powi, sqrt};

/// Takes the interval and splits it into n sub-intervals.
/// Returns the resulting n+1 boundary points.
pub fn make_supporting_points(n: usize, interval: Interval) -> Vec<f64> {
    let h = interval.span() / n as f64;
    let mut pts: Vec<f64> = (0..n)
        .map(|step| interval.start() + step as f64 * h)
        .collect();
    pts.push(interval.end());
    pts
}

/// Samples the function at `n_samples` equidistant points on the interval.
pub fn sample_closure<T: Copy>(
    f: Closure1D<T>,
    interval: Interval,
    n_samples: usize,
    data: T,
) -> Vec<Point2D> {
    make_supporting_points(n_samples - 1, interval)
        .iter()
        .map(|x| Point2D {
            x: *x,
            y: f(*x, data),
        })
        .collect()
}

/// Samples the function at `n_samples` equidistant points on the interval.
pub fn sample_function(f: Function1D, interval: Interval, n_samples: usize) -> Vec<Point2D> {
    make_supporting_points(n_samples, interval)
        .iter()
        .map(|x| Point2D { x: *x, y: f(*x) })
        .collect()
}

/// Samples the function at `n_samples` equidistant points on the interval.
/// Returns vector of results.
pub fn sample_function_generic<T>(f: fn(f64) -> T, interval: Interval, n_samples: usize) -> Vec<T> {
    make_supporting_points(n_samples, interval)
        .iter()
        .map(|t| f(*t))
        .collect()
}

/// Calculates the euclidean norm of a vector.
pub fn euclidean_norm(v: Vec<f64>) -> f64 {
    sqrt!(v.iter().map(|x| x * x).sum::<f64>())
}

/// Specialty function for task 1.
/// Could be made more general.
pub fn get_convergence_order(abs_errors: &[f64], hs: &[f64]) -> f64 {
    let ps = get_all_convergence_orders(abs_errors, hs);

    ps.iter().sum::<f64>() / ps.len() as f64
}

/// Calculates the numeric convergence order for every two consecutive samples.
pub fn get_all_convergence_orders(abs_errors: &[f64], hs: &[f64]) -> Vec<f64> {
    abs_errors
        .iter()
        .zip(hs)
        .zip(abs_errors.iter().skip(1))
        .zip(hs.iter().skip(1))
        .map(|(((abs_error, h1), abs_error2), h2)| {
            (ln!(abs_error2) - ln!(abs_error)) / (ln!(h2) - ln!(h1))
        })
        .collect()
}

/// Make a vector of the given length only containing zeroes.
pub fn make_zero_vec(len: usize) -> Vec<f64> {
    (0..len).map(|_| 0.0).collect()
}

/// Computes the sum of squared errors between points
pub fn sse(v1: &[Point2D], v2: &[Point2D]) -> f64 {
    v1.iter()
        .zip(v2.iter())
        .map(|(p1, p2)| powi!(p1.x - p2.x, 2) + powi!(p1.y - p2.y, 2))
        .sum()
}

/// abs(x)
#[macro_export]
macro_rules! abs {
    ($name: expr) => {
        $name.abs()
    };
}

/// sqrt(x)
#[macro_export]
macro_rules! sqrt {
    ($name: expr) => {
        $name.sqrt()
    };
}

/// ln(x)
#[macro_export]
macro_rules! ln {
    ($name: expr) => {
        $name.ln()
    };
}

/// x^y with y some kind of float
#[macro_export]
macro_rules! powf {
    ($name: expr, $exponent: expr) => {
        $name.powf($exponent)
    };
}

/// x^n with n some kind of int
#[macro_export]
macro_rules! powi {
    ($name: expr, $exponent: expr) => {
        $name.powi($exponent)
    };
}

/// sin(x)
#[macro_export]
macro_rules! sin {
    ($name: expr) => {
        $name.sin()
    };
}

/// cos(x)
#[macro_export]
macro_rules! cos {
    ($name: expr) => {
        $name.cos()
    };
}

/// ceiling(x)
#[macro_export]
macro_rules! ceil {
    ($name: expr) => {
        $name.ceil() as usize
    };
}

/// e^x
#[macro_export]
macro_rules! exp {
    ($name: expr) => {
        E.powf($name)
    };
}
