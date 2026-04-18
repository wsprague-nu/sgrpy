//! Enum for Ac factor calculation and estimation methods

pub enum AcCalculation {}

pub enum AcEstimation {}

impl AcEstimation {
    pub fn compatible_with(&self, calculation: &AcCalculation) -> bool {
        false
    }
}
