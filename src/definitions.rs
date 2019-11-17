use crate::abs;
use derive_new::*;
use std::fmt::{Display, Error, Formatter};
use std::marker::PhantomData;
use std::ops::Mul;

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

/// Type alias for a function of t x R^n -> R
pub type FunctionND<T, R> = fn(T) -> R;

/// Type alias
pub type SimpleDifferentiableFunction2D = SimpleDifferentiableFunction<(f64, f64)>;
/// Type alias
pub type SimpleDifferentiableFunction1D = SimpleDifferentiableFunction<f64>;

impl<T, R> SampleableFunction<T, R> for FunctionND<T, R> {
    fn value_at(&self, input: T) -> R {
        self(input)
    }
}

/// It's impossible to implement the Sub trait for Vec<f64>.
/// Because of that I re-created it shortly.
pub trait PointwiseSub {
    /// Subtraction
    fn pointwise_sub(self, rhs: Self) -> Self;
}

impl PointwiseSub for Vec<f64> {
    fn pointwise_sub(self, rhs: Self) -> Self {
        self.iter().zip(rhs).map(|(a, b)| a - b).collect()
    }
}

/// Pointwise addition for vectors
pub trait PointwiseAdd {
    /// Addition
    fn pointwise_add(self, rhs: Self) -> Self;
}

impl PointwiseAdd for Vec<f64> {
    fn pointwise_add(self, rhs: Self) -> Self {
        self.iter().zip(rhs).map(|(a, b)| a + b).collect()
    }
}

/// Scalar multiplication for vectors
pub trait ScalarMul<T: Mul> {
    /// Addition
    fn scalar_mul(self, rhs: T) -> Self;
}

impl ScalarMul<f64> for Vec<f64> {
    fn scalar_mul(self, rhs: f64) -> Self {
        self.iter().map(|a| a * rhs).collect()
    }
}

/// Implementation of DifferentiableFunction that uses Function<T>
#[derive(Copy, Clone, Debug, new)]
pub struct SimpleDifferentiableFunction<T> {
    /// f
    pub f: Function<T>,
    /// f'
    pub df: Function<T>,
}

impl<T> SampleableFunction<T, f64> for SimpleDifferentiableFunction<T> {
    fn value_at(&self, input: T) -> f64 {
        (self.f)(input)
    }
}

impl<T> DifferentiableFunction<T, f64> for SimpleDifferentiableFunction<T> {
    fn derivative_at(&self, input: T) -> f64 {
        (self.df)(input)
    }
}

/// Implementation for SampleableFunction that uses Closure<T>.
#[derive(new, Clone)]
pub struct ClosureSampleableFunction<T, Data: Copy> {
    data: Data,
    f: Closure<T, Data>,
}

impl<T, Data: Copy> SampleableFunction<T, f64> for ClosureSampleableFunction<T, Data> {
    fn value_at(&self, input: T) -> f64 {
        (self.f)(input, self.data)
    }
}

/// Implementation for DifferentiableFunction that uses Closure<T>.
#[derive(new)]
pub struct ClosureDifferentiableFunction<T, Data: Copy> {
    data: Data,
    f: Closure<T, Data>,
    df: Closure<T, Data>,
}

impl<T, Data: Copy> SampleableFunction<T, f64> for ClosureDifferentiableFunction<T, Data> {
    fn value_at(&self, input: T) -> f64 {
        (self.f)(input, self.data)
    }
}

impl<T, Data: Copy> DifferentiableFunction<T, f64> for ClosureDifferentiableFunction<T, Data> {
    fn derivative_at(&self, input: T) -> f64 {
        (self.df)(input, self.data)
    }
}

/// Mathematical composition of functions for my SampleableFunction trait.
/// You basically give it a f_inner and f_outer and it does f_outer . f_inner.
#[derive(new)]
pub struct ComposeSampleableFunction<
    T,
    R,
    R2,
    FT: SampleableFunction<T, R>,
    FT2: SampleableFunction<R, R2>,
> {
    _t: PhantomData<T>,
    _r: PhantomData<R>,
    _r2: PhantomData<R2>,
    inner: FT,
    outer: FT2,
}

impl<T, R, R2, FT: SampleableFunction<T, R>, FT2: SampleableFunction<R, R2>>
    SampleableFunction<T, R2> for ComposeSampleableFunction<T, R, R2, FT, FT2>
{
    fn value_at(&self, input: T) -> R2 {
        self.outer.value_at(self.inner.value_at(input))
    }
}

/// For two sampleable functions this returns the difference between all first and second values.
#[derive(new)]
pub struct SubSampleableFunction<
    T,
    R: PointwiseSub,
    FT: SampleableFunction<T, R>,
    FT2: SampleableFunction<T, R>,
> {
    _t: PhantomData<T>,
    _r: PhantomData<R>,
    a: FT,
    b: FT2,
}

impl<T: Clone, R: PointwiseSub, FT: SampleableFunction<T, R>, FT2: SampleableFunction<T, R>>
    SampleableFunction<T, R> for SubSampleableFunction<T, R, FT, FT2>
{
    fn value_at(&self, input: T) -> R {
        self.a
            .value_at(input.clone())
            .pointwise_sub(self.b.value_at(input.clone()))
    }
}

/// For two sampleable functions this returns the product of all first and second values.
#[derive(new)]
pub struct MultSampleableFunction<
    T,
    R: Mul<Output = R>,
    FT: SampleableFunction<T, R>,
    FT2: SampleableFunction<T, R>,
> {
    _t: PhantomData<T>,
    _r: PhantomData<R>,
    a: FT,
    b: FT2,
}

impl<T: Clone, R: Mul<Output = R>, FT: SampleableFunction<T, R>, FT2: SampleableFunction<T, R>>
    SampleableFunction<T, R> for MultSampleableFunction<T, R, FT, FT2>
{
    fn value_at(&self, input: T) -> R {
        self.a
            .value_at(input.clone())
            .mul(self.b.value_at(input.clone()))
    }
}

/// Takes a differentiable function and returns the values of its derivative on being sampled.
#[derive(new)]
pub struct SampledDerivative<T, R, FT: DifferentiableFunction<T, R>> {
    _t: PhantomData<T>,
    _r: PhantomData<R>,
    f: FT,
}

impl<T, R, FT: DifferentiableFunction<T, R>> SampleableFunction<T, R>
    for SampledDerivative<T, R, FT>
{
    fn value_at(&self, input: T) -> R {
        self.f.derivative_at(input)
    }
}

/// Describes a one time differentiable f -> R.
/// Note that continuity of the function is not enforced and let to the user.
pub trait DifferentiableFunction<T, R>: SampleableFunction<T, R> {
    /// f'(input)
    fn derivative_at(&self, input: T) -> R;
}

/// Describes a function -> R that can be sampled at every point of T (whatever that currently is).
/// That does not mean it is continuous!
pub trait SampleableFunction<T, R> {
    /// f(input)
    fn value_at(&self, input: T) -> R;
}

/// A simple point on the x/y plane.
/// TODO maybe generalize
#[derive(Copy, Clone, Debug, new)]
pub struct Point2D {
    /// x coordinate
    pub x: f64,
    /// y coordinate
    pub y: f64,
}

/// Inclusive interval of the form [a, b] < R
#[derive(Copy, Clone, Debug, new)]
pub struct Interval {
    start: f64,
    end: f64,
}

impl Interval {
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
#[derive(Copy, Clone, Debug, new)]
pub struct InitialValueProblem<FT> {
    /// t_0
    pub start_time: f64,
    /// f(t_0)
    pub start_value: f64,
    /// f'(x)
    pub df: FT,
}

impl InitialValueProblem<Function2D> {
    /// Helper to make multi dimensional euler usable for one dimensional problems
    pub fn to_system_problem(
        self,
    ) -> InitialValueSystemProblem<ClosureSampleableFunction<(f64, Vec<f64>), Function2D>> {
        let new_df: Closure<(f64, Vec<f64>), Function2D> = |(t, vec_x), df| df((t, vec_x[0]));
        InitialValueSystemProblem::new(
            self.start_time,
            vec![self.start_value],
            vec![ClosureSampleableFunction::new(self.df, new_df)],
        )
    }
}

/// Assume that values and dfs are same size and functions match...
#[derive(Clone, Debug, new)]
pub struct InitialValueSystemProblem<FT: SampleableFunction<(f64, Vec<f64>), f64>> {
    /// t_0
    pub start_time: f64,
    /// f(t_0)
    pub start_values: Vec<f64>,
    /// f'(x)
    pub dfs: Vec<FT>,
}

/// Problem like in task 2 subtask 4.
/// Likely will become more complex over time.
#[derive(Copy, Clone, Debug, new)]
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
