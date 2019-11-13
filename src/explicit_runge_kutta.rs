use derive_new::*;

#[derive(Clone, Debug, new)]
pub struct Tableau {
    cs: Vec<f64>,
    bs: Vec<f64>,
    coeffs: Vec<Vec<f64>>
}

#[derive(Clone, Debug, new)]
pub struct ExplicitRungeKuttaMethod {
    tableau: Tableau
}