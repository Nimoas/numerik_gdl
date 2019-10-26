use crate::definitions::Interval;

pub fn make_supporting_points(n: usize, interval: Interval) -> Vec<f64> {
    let h = interval.span() / n as f64;
    let mut pts: Vec<f64> = (0..n)
        .map(|step| interval.start() + step as f64 * h)
        .collect();
    pts.push(interval.end());
    pts
}
