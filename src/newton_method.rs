use crate::abs;
use crate::definitions::{DifferentiableFunction, SampleableFunction};

pub fn newton_method<F: DifferentiableFunction<(f64, f64), f64>>(
    func: F,
    t: f64,
    start_x: f64,
    eps: f64,
) -> f64 {
    // Don't do work if our guess is good enough already...
    if abs!(func.value_at((t, start_x))) < eps {
        return start_x;
    }

    let mut last = start_x;
    let mut current = newton_step(&func, t, last);

    while abs!(func.value_at((t, current))) > eps {
        last = current;
        current = newton_step(&func, t, last);
    }

    current
}

fn newton_step<F: DifferentiableFunction<(f64, f64), f64>>(func: &F, t: f64, val: f64) -> f64 {
    val - (func.value_at((t, val)) / func.derivative_at((t, val)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::definitions::SimpleDifferentiableFunction;

    #[test]
    fn test_simple() {
        let func = SimpleDifferentiableFunction::new(|(_, x)| x * x, |(_, x)| 2.0 * x);
        let eps = 0.001;

        let func_val = func.value_at((0.0, newton_method(func, 0.0, 2.0, eps)));
        assert!(eps > abs!(func_val));
    }
}
