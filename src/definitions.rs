use std::fmt::{Display, Formatter, Error};

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