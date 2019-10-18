pub type Function1D = fn(x: f64) -> f64;

pub struct Interval {
    start: f64,
    end: f64,
}

impl Interval {
    pub fn new(a: f64, b: f64) -> Self {
        Interval {
            start: a,
            end: b,
        }
    }

    pub fn start(&self) -> f64 {
        self.start
    }

    pub fn end(&self) -> f64 {
        self.end
    }

    pub fn span(&self) -> f64 {
        (self.start - self.end).abs()
    }
}