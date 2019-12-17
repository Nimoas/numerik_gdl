use crate::definitions::BoundaryValueProblem;
use crate::util::make_supporting_points;
use nalgebra::DMatrix;
use rayon::prelude::*;

// TODO version of solve_bvp that returns step function?

/// Rather specific implementation of a finite differences method for task 2 subtask 4.
/// Probably will be generalized in one of the next tasks?
///
///# Arguments
///
/// * `problem` - The boundary value problem to solve.
/// * `n_grid` - Number of sample points (not including the boundary points!). From this results h = interval.span / (n_grid + 1)
///
/// # Example
/// ```
/// use ngdl_rust::definitions::{Interval, BoundaryValueProblem};
/// use ngdl_rust::finite_differences_method::solve_bvp;
///
/// let interval = Interval::new(0.0, 1.0);
/// let bvp = BoundaryValueProblem::new(|x| x.sqrt(), interval, 1.0, 0.0);
/// let n_grid = 99;
///
/// // Calling the method to get a solution. Could be unsolvable -> expect with error message.
/// let solution = solve_bvp(bvp, n_grid).expect("No solution found!");
/// ```
pub fn solve_bvp(problem: BoundaryValueProblem, n_grid: usize) -> Option<Vec<f64>> {
    // Setting up the sampled points
    let h = problem.interval.span() / (n_grid as f64 + 1.0);
    let grid_x_values = &make_supporting_points(n_grid + 1, problem.interval)[1..=n_grid];

    // Setting up the right hand side of the equation system.
    let mut right_side: Vec<f64> = grid_x_values
        .par_iter()
        .map(|x| (problem.ddf)(*x))
        .map(|x| x * h * h)
        .collect();
    right_side[0] -= problem.start_value;
    right_side[n_grid - 1] -= problem.end_value;
    let to_solve = DMatrix::from_vec(n_grid, 1, right_side);

    // Build a matrix with a 1 -2 1 pattern down its diagonal
    // !!! This is only right for working with the central difference formula for the 2nd derivative.
    let matrix: DMatrix<f64> = DMatrix::from_fn(n_grid, n_grid, |row, col| {
        if row == col {
            return -2.0;
        }
        if row + 1 == col || col + 1 == row {
            return 1.0;
        }
        0.0
    });

    // Solving for fun and profit.
    // If we ever need more calculation power, one can always use the lib with lapack.
    let decomposition = matrix.lu();
    decomposition
        .solve(&to_solve)
        .map(|mat| mat.iter().copied().collect())
}
