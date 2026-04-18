//! Calculation options struct

use super::{AcCalculation, AsCalculation, SplitPolicy};

pub struct CalcOptions {
    pub n_count: Option<usize>,
    pub p_cover: Option<f64>,
    pub split_policy: SplitPolicy,
    pub ac_calculation: Option<AcCalculation>,
    pub as_calculation: Option<AsCalculation>,
}
