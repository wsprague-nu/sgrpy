use itertools::Itertools;
use ordered_float::OrderedFloat;

use super::super::links::{ColumnSelect, DancingLinks};
use super::utils::calc_entropy;

/// Column selector
#[derive(Debug)]
pub struct ColSelector {
    costs: Vec<f64>,
}

impl ColSelector {
    pub fn from_costs(costs: impl IntoIterator<Item = f64>) -> Self {
        ColSelector {
            costs: costs.into_iter().collect(),
        }
    }

    pub fn select_col(&self, links: &DancingLinks) -> ColumnSelect {
        match links.select_column() {
            ColumnSelect::Column(_) => {
                let active_cols = links.get_active_cols();
                let result = active_cols.into_iter().min_by_key(|&c| {
                    OrderedFloat(self.calc_entropy_of_rows(links.get_rows(c)))
                });
                if let Some(result_col) = result {
                    ColumnSelect::Column(result_col)
                } else {
                    ColumnSelect::Fail
                }
            }
            ColumnSelect::Complete => ColumnSelect::Complete,
            ColumnSelect::Fail => ColumnSelect::Fail,
        }
    }

    fn calc_entropy_of_rows(&self, rows: Vec<usize>) -> f64 {
        calc_entropy(&rows.into_iter().map(|r| self.costs[r]).collect_vec())
    }
}
