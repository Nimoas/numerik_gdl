use crate::definitions::{Interval, Point2D, SampleableFunction};
use crate::util::make_supporting_points;
use itertools::iproduct;
use num::complex::Complex64;

/// Sampling whole area instead of contours, because I'm lazy and don't have that as a builting unlike octave.
pub fn sample_stability_area<FT: SampleableFunction<Complex64, f64>>(
    f: FT,
    n_samples: usize,
    re_interval: Interval,
    im_interval: Interval,
) -> Vec<Point2D> {
    println!(
        "Sampling stability area. Projected samples: {}",
        n_samples * n_samples
    );

    let re_samples = make_supporting_points(n_samples, re_interval);
    let im_samples = make_supporting_points(n_samples, im_interval);

    iproduct!(re_samples, im_samples)
        .map(|(r, i)| {
            let z = Complex64::new(r, i);
            (z, f.value_at(z))
        })
        .filter(|(_, v)| *v <= 1.0)
        .map(|(z, _)| Point2D::new(z.re, z.im))
        .collect()
}
