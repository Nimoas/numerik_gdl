use crate::abs;
use std::fmt::{Display, Error, Formatter};

/// Type alias for a R -> R function.
pub type Function1D = Function<f64>;
pub type Closure1D<T> = Closure<f64, T>;
pub type Function<T> = fn(T) -> f64;
pub type Closure<T, Data> = fn(T, Data) -> f64;

/// A type alias for a function of R x R -> R.
/// Generally used for the f(t, x_t) part of DGLs.
pub type Function2D = Function<(f64, f64)>;
pub type Closure2D<T> = Closure<(f64, f64), T>;

pub type SimpleDifferentiableFunction2D = SimpleDifferentiableFunction<Function2D>;

#[derive(Copy, Clone, Debug)]
pub struct SimpleDifferentiableFunction<T> {
    pub f: Function<T>,
    pub df: Function<T>,
}

impl<T> SimpleDifferentiableFunction<T> {
    pub fn new(f: Function<T>, df: Function<T>) -> SimpleDifferentiableFunction<T> {
        SimpleDifferentiableFunction { f, df }
    }
}

impl<T> DifferentiableFunction<T> for SimpleDifferentiableFunction<T> {
    fn value_at(&self, input: T) -> f64 {
        (self.f)(input)
    }

    fn derivative_at(&self, input: T) -> f64 {
        (self.df)(input)
    }
}

pub struct ClosureDifferentiableFunction<T, Data: Copy> {
    data: Data,
    f: Closure<T, Data>,
    df: Closure<T, Data>,
}

impl<T, Data: Copy> ClosureDifferentiableFunction<T, Data> {
    pub fn new(
        data: Data,
        f: Closure<T, Data>,
        df: Closure<T, Data>,
    ) -> ClosureDifferentiableFunction<T, Data> {
        ClosureDifferentiableFunction { data, f, df }
    }
}

impl<T, Data: Copy> DifferentiableFunction<T> for ClosureDifferentiableFunction<T, Data> {
    fn value_at(&self, input: T) -> f64 {
        (self.f)(input, self.data)
    }

    fn derivative_at(&self, input: T) -> f64 {
        (self.df)(input, self.data)
    }
}

pub trait DifferentiableFunction<T> {
    fn value_at(&self, input: T) -> f64;
    fn derivative_at(&self, input: T) -> f64;
}

#[derive(Copy, Clone, Debug)]
pub struct Point2D {
    pub x: f64,
    pub y: f64,
}

#[derive(Copy, Clone, Debug)]
pub struct Interval {
    start: f64,
    end: f64,
}

impl Interval {
    pub fn new(a: f64, b: f64) -> Self {
        Interval { start: a, end: b }
    }

    pub fn start(&self) -> f64 {
        self.start
    }

    pub fn end(&self) -> f64 {
        self.end
    }

    /// Returns the absolute difference between the start and end of the interval.
    pub fn span(&self) -> f64 {
        abs!(self.start - self.end)
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
pub struct InitialValueProblem<FT> {
    pub start_time: f64,
    pub start_value: f64,
    pub df: FT,
}

impl<FT> InitialValueProblem<FT> {
    pub fn new(start_time: f64, start_value: f64, df: FT) -> InitialValueProblem<FT> {
        InitialValueProblem {
            start_time,
            start_value,
            df,
        }
    }
}

/// Problem like in task 2 subtask 4.
/// Likely will become more complex over time.
#[derive(Copy, Clone, Debug)]
pub struct BoundaryValueProblem {
    pub ddf: Function1D,
    pub interval: Interval,
    pub start_value: f64,
    pub end_value: f64,
}

impl BoundaryValueProblem {
    pub fn new(
        ddf: Function1D,
        interval: Interval,
        start_value: f64,
        end_value: f64,
    ) -> BoundaryValueProblem {
        BoundaryValueProblem {
            ddf,
            interval,
            start_value,
            end_value,
        }
    }
}
