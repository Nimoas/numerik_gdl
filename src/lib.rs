#![warn(missing_docs)]

//! My implementations of the tasks for "Numerik gewöhnlicher Differentialgleichungen"

/// Contains generic definitions and helper methods
pub mod definitions;
/// Implementation of the explicit euler method
pub mod euler_explicit;
/// Basic implementation of an explicit Runge-Kutta method.
pub mod explicit_runge_kutta;
/// Implementation of the finite differences method
pub mod finite_differences_method;
mod generalized_explicit_one_step_method;
/// Implementation of the implicit euler method
pub mod implicit_euler;
/// Explicit euler also using the derivative of the given DGL.
pub mod modified_explicit_euler;
pub mod adams_bashfoth;
mod newton_method;
/// Little plot helpers to reduce boilerplate
pub mod plot_util;
/// Numeric quadrature with several methods
pub mod quadrature;
/// Helpful helpers for common computations
pub mod util;
