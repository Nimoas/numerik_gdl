use crate::definitions::{Closure1D, Function1D, Interval, Point2D};

pub fn make_supporting_points(n: usize, interval: Interval) -> Vec<f64> {
    let h = interval.span() / n as f64;
    let mut pts: Vec<f64> = (0..n)
        .map(|step| interval.start() + step as f64 * h)
        .collect();
    pts.push(interval.end());
    pts
}

pub fn sample_closure<T: Copy>(
    f: Closure1D<T>,
    interval: Interval,
    n_samples: usize,
    data: T,
) -> Vec<Point2D> {
    make_supporting_points(n_samples, interval)
        .iter()
        .map(|x| Point2D {
            x: *x,
            y: f(*x, data),
        })
        .collect()
}

pub fn sample_function(f: Function1D, interval: Interval, n_samples: usize) -> Vec<Point2D> {
    make_supporting_points(n_samples, interval)
        .iter()
        .map(|x| Point2D { x: *x, y: f(*x) })
        .collect()
}

#[macro_export]
macro_rules! abs {
    ($name: expr) => {
        $name.abs()
    };
}

#[macro_export]
macro_rules! sqrt {
    ($name: expr) => {
        $name.sqrt()
    };
}

#[macro_export]
macro_rules! ln {
    ($name: expr) => {
        $name.ln()
    };
}

#[macro_export]
macro_rules! powf {
    ($name: expr, $exponent: expr) => {
        $name.powf($exponent)
    };
}

#[macro_export]
macro_rules! powi {
    ($name: expr, $exponent: expr) => {
        $name.powi($exponent)
    };
}

#[macro_export]
macro_rules! sin {
    ($name: expr) => {
        $name.sin()
    };
}

#[macro_export]
macro_rules! cos {
    ($name: expr) => {
        $name.cos()
    };
}

#[macro_export]
macro_rules! exp {
    ($name: expr) => {
        E.powf($name)
    };
}
