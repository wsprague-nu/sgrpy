use itertools::Itertools;

#[derive(Debug)]
pub struct MinEstAvg {
    costs: Vec<(f64, Vec<usize>)>,
    num_cols: usize,
}

impl MinEstAvg {
    pub fn from_examples(
        iter: impl IntoIterator<Item = (impl IntoIterator<Item = usize>, f64)>,
    ) -> Self {
        let cost_vec: Vec<(f64, Vec<usize>)> = iter
            .into_iter()
            .map(|(data, weight)| (weight, data.into_iter().collect_vec()))
            .map(|(weight, datavec)| (weight / (datavec.len() as f64), datavec))
            .collect();
        let max_col: usize = cost_vec
            .iter()
            .map(|(_, datavec)| *datavec.iter().max().unwrap_or(&0))
            .max()
            .unwrap_or(0);
        MinEstAvg {
            costs: cost_vec,
            num_cols: max_col + 1,
        }
    }

    pub fn min_value(&self, rows: &Vec<usize>) -> (f64, f64) {
        let mut covered_rows = std::collections::HashSet::new();
        let mut current_weight: Vec<f64> = Vec::new();
        for row in rows {
            let (row_weight, cols) = &self.costs[*row];
            current_weight.push(*row_weight);
            for &col in cols.iter() {
                covered_rows.insert(col);
            }
        }
        let mut remaining_costs: Vec<Option<f64>> = vec![None; self.num_cols];
        for (weight, rows) in self.costs.iter() {
            if rows.iter().any(|x| covered_rows.contains(x)) {
                continue;
            }
            for &x in rows.iter() {
                if let Some(cur_weight) = remaining_costs[x] {
                    if cur_weight > *weight {
                        remaining_costs[x] = Some(*weight);
                    }
                } else {
                    remaining_costs[x] = Some(*weight);
                }
            }
        }
        let cur_cost = current_weight.into_iter().sum();
        let fut_cost = remaining_costs.into_iter().flatten().sum();
        (cur_cost, fut_cost)
    }
}
