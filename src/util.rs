use crate::definitions::{Closure1D, Function1D, Interval, Point2D};
use crate::sqrt;

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

/// Calculates the euclidean norm of a vector.
pub fn euclidean_norm(v: Vec<f64>) -> f64 {
    sqrt!(v.iter().map(|x| x * x).sum::<f64>())
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
