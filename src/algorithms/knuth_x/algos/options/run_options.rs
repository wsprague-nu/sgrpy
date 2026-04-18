//! Runtime options struct

use super::{AcEstimation, AsEstimation, CostEstimation};

pub struct RunOptions {
    pub max_tree: Option<usize>,
    pub max_cost: Option<f64>,
    pub cost_estimator: CostEstimation,
    pub ac_estimator: AcEstimation,
    pub as_estimator: AsEstimation,
}
