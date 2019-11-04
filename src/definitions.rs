use crate::abs;
use std::fmt::{Display, Error, Formatter};

/// Generic type for a function -> R
/// T can be e.g. f64, or (f64, f64).
pub type Function<T> = fn(T) -> f64;
/// Generic type for a function -> R that also takes additional data.
/// This is necessary because constructing dynamic closures plays havoc with the Rust borrow checker...
pub type Closure<T, Data> = fn(T, Data) -> f64;

/// Type alias for a R -> R function.
pub type Function1D = Function<f64>;
/// Type alias
pub type Closure1D<T> = Closure<f64, T>;

/// A type alias for a function of R x R -> R.
/// Generally used for the f(t, x_t) part of DGLs.
pub type Function2D = Function<(f64, f64)>;
/// Type alias
pub type Closure2D<T> = Closure<(f64, f64), T>;

/// Type alias
pub type SimpleDifferentiableFunction2D = SimpleDifferentiableFunction<Function2D>;
/// Type alias
pub type SimpleDifferentiableFunction1D = SimpleDifferentiableFunction<Function1D>;

/// Implementation of DifferentiableFunction that uses Function<T>
#[derive(Copy, Clone, Debug)]
pub struct SimpleDifferentiableFunction<T> {
    /// f
    pub f: Function<T>,
    /// f'
    pub df: Function<T>,
}

impl<T> SimpleDifferentiableFunction<T> {
    /// Create a new function from f and f'
    pub fn new(f: Function<T>, df: Function<T>) -> SimpleDifferentiableFunction<T> {
        SimpleDifferentiableFunction { f, df }
    }
}

impl<T> SampleableFunction<T> for SimpleDifferentiableFunction<T> {
    fn value_at(&self, input: T) -> f64 {
        (self.f)(input)
    }
}

impl<T> DifferentiableFunction<T> for SimpleDifferentiableFunction<T> {
    fn derivative_at(&self, input: T) -> f64 {
        (self.df)(input)
    }
}

/// Implementation for DifferentiableFunction that uses Closure<T>.
pub struct ClosureDifferentiableFunction<T, Data: Copy> {
    data: Data,
    f: Closure<T, Data>,
    df: Closure<T, Data>,
}

impl<T, Data: Copy> ClosureDifferentiableFunction<T, Data> {
    /// Create a new function with the passed data used in the closure
    pub fn new(
        data: Data,
        f: Closure<T, Data>,
        df: Closure<T, Data>,
    ) -> ClosureDifferentiableFunction<T, Data> {
        ClosureDifferentiableFunction { data, f, df }
    }
}

impl<T, Data: Copy> SampleableFunction<T> for ClosureDifferentiableFunction<T, Data> {
    fn value_at(&self, input: T) -> f64 {
        (self.f)(input, self.data)
    }
}

impl<T, Data: Copy> DifferentiableFunction<T> for ClosureDifferentiableFunction<T, Data> {
    fn derivative_at(&self, input: T) -> f64 {
        (self.df)(input, self.data)
    }
}

/// Describes a one time differentiable f -> R.
/// Note that continuity of the function is not enforced and let to the user.
pub trait DifferentiableFunction<T>: SampleableFunction<T> {
    /// f'(input)
    fn derivative_at(&self, input: T) -> f64;
}

/// Describes a function -> R that can be sampled at every point of T (whatever that currently is).
/// That does not mean it is continuous!
pub trait SampleableFunction<T> {
    /// f(input)
    fn value_at(&self, input: T) -> f64;
}

/// A simple point on the x/y plane.
/// TODO maybe generalize
#[derive(Copy, Clone, Debug)]
pub struct Point2D {
    /// x coordinate
    pub x: f64,
    /// y coordinate
    pub y: f64,
}

/// Inclusive interval of the form [a, b] < R
#[derive(Copy, Clone, Debug)]
pub struct Interval {
    start: f64,
    end: f64,
}

impl Interval {
    /// Create an interval
    pub fn new(a: f64, b: f64) -> Self {
        Interval { start: a, end: b }
    }

    /// Get first value in interval
    pub fn start(&self) -> f64 {
        self.start
    }

    /// Get last value in interval
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
/// TODO enforce trait bound of SampleableFunction for FT
#[derive(Copy, Clone, Debug)]
pub struct InitialValueProblem<FT> {
    /// t_0
    pub start_time: f64,
    /// f(t_0)
    pub start_value: f64,
    /// f'(x)
    pub df: FT,
}

impl<FT> InitialValueProblem<FT> {
    /// Construct a new IVP
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
    /// f''(x)
    pub ddf: Function1D,
    /// Inclusive interval on which we want to approximate f(x)
    pub interval: Interval,
    /// f(interval.start())
    pub start_value: f64,
    /// f(interval.end())
    pub end_value: f64,
}

impl BoundaryValueProblem {
    /// Construct a new BVP
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
