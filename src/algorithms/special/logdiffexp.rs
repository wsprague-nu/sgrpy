use std::cmp::Ordering::{Equal, Greater, Less};

/// Compute the log of the difference of exponentials of input arguments.
pub fn logdiffexp(a: f64, b: f64) -> f64 {
    match a.partial_cmp(&b) {
        Some(Greater) => a + (-(b - a).exp()).ln_1p(),
        Some(Equal) => f64::NEG_INFINITY,
        Some(Less) | None => f64::NAN,
    }
}
