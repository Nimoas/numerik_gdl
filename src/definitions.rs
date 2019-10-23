use std::fmt::{Display, Formatter, Error};

/// Type alias for a R -> R function.
pub type Function1D = fn(x: f64) -> f64;

/// A type alias for a function of R x R -> R.
/// Generally used for the f(t, x_t) part of DGLs.
pub type Function2D = fn(f64, f64) -> f64;

#[derive(Copy, Clone, Debug)]
pub struct Interval {
    start: f64,
    end: f64,
}

impl Interval {
    pub fn new(a: f64, b: f64) -> Self {
        Interval {
            start: a,
            end: b,
        }
    }

    pub fn start(&self) -> f64 {
        self.start
    }

    pub fn end(&self) -> f64 {
        self.end
    }

    /// Returns the absolute difference between the start and end of the interval.
    pub fn span(&self) -> f64 {
        (self.start - self.end).abs()
    }
}

impl Display for Interval {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        f.write_str("[")?;
        f.write_str(&self.start().to_string())?;
        f.write_str(", ")?;
        f.write_str(&self.end().to_string())?;
        f.write_str("]")?;
        Ok(())
    }
}

/// Representing an initial value problem on R.
/// Build to encapsulate the data fed into e.g. euler's method.
#[derive(Copy, Clone, Debug)]
pub struct InitialValueProblem {
    pub start_time: f64,
    pub start_value: f64,
    pub df: Function2D
}

impl InitialValueProblem {
    pub fn new(start_time: f64,
               start_value: f64,
               df: Function2D) -> InitialValueProblem {
        InitialValueProblem {
            start_time,
            start_value,
            df
        }
    }
}