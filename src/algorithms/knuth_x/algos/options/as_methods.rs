//! Enum for As factor calculation and estimation methods

pub enum AsCalculation {}

pub enum AsEstimation {}

impl AsEstimation {
    pub fn compatible_with(&self, calculation: &AsCalculation) -> bool {
        false
    }
}
