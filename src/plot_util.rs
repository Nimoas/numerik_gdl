use crate::definitions::Point2D;
use gnuplot::{Axes2D, PlotOption};

/// Plots a number of points as line on the given axis.
pub fn plot_line_on(axis: &mut Axes2D, points: &[Point2D], options: &[PlotOption<&str>]) {
    axis.lines(
        &points.iter().map(|p| p.x).collect::<Vec<f64>>(),
        &points.iter().map(|p| p.y).collect::<Vec<f64>>(),
        options,
    );
}

/// Plots a number of points as line with markers on the given axis.
pub fn plot_line_points_on(axis: &mut Axes2D, points: &[Point2D], options: &[PlotOption<&str>]) {
    axis.lines_points(
        &points.iter().map(|p| p.x).collect::<Vec<f64>>(),
        &points.iter().map(|p| p.y).collect::<Vec<f64>>(),
        options,
    );
}

/// Plots a number of points the given axis.
pub fn plot_points_on(axis: &mut Axes2D, points: &[Point2D], options: &[PlotOption<&str>]) {
    axis.points(
        &points.iter().map(|p| p.x).collect::<Vec<f64>>(),
        &points.iter().map(|p| p.y).collect::<Vec<f64>>(),
        options,
    );
}
