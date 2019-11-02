use crate::definitions::Interval;

pub fn make_supporting_points(n: usize, interval: Interval) -> Vec<f64> {
    let h = interval.span() / n as f64;
    let mut pts: Vec<f64> = (0..n)
        .map(|step| interval.start() + step as f64 * h)
        .collect();
    pts.push(interval.end());
    pts
}

#[macro_export]
macro_rules! abs {
    ($name: expr) => {
        $name.abs()
    };
}

#[macro_export]
macro_rules! ln {
    ($name: expr) => {
        $name.ln()
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
