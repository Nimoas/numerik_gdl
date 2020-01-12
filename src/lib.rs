#![warn(missing_docs)]

//! My implementations of the tasks for "Numerik gewöhnlicher Differentialgleichungen"

/// Implementations for the explicit Adams Bashforth methods.
pub mod adams_bashforth;
mod constants;
/// Contains generic definitions and helper methods
pub mod definitions;
/// Implementation of the explicit euler method
pub mod euler_explicit;
/// Basic implementation of an explicit Runge-Kutta method.
pub mod explicit_runge_kutta;
/// Implementation of the finite differences method
pub mod finite_differences_method;
mod generalized_explicit_k_step_method;
mod generalized_explicit_one_step_method;
/// There be dragons. Explicit versions of implicit methods.
pub mod hack;
/// Implementation of the implicit euler method
pub mod implicit_euler;
/// Implementation of the Milne Simpson predictor-corrector method.
pub mod milne_simpson;
/// Explicit euler also using the derivative of the given DGL.
pub mod modified_explicit_euler;
mod newton_method;
/// Explicit Nyström method
pub mod nystroem;
/// Little plot helpers to reduce boilerplate
pub mod plot_util;
/// Numeric quadrature with several methods
pub mod quadrature;
/// Functions to sample stability functions to get stability areas.
pub mod stability_area;
/// Helpful helpers for common computations
pub mod util;

/// Re-export constants at top level.
pub use constants::*;
