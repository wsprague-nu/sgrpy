use std::collections::HashSet;

use super::get_top::get_top_covers;
use crate::algorithms::special::{logdiffexp, logsumexp};

/// estimates the Bell number of n
fn log_bell_number(n: f64) -> f64 {
    n * ((0.792 * n) / (n + 1.).ln()).ln()
}

struct CurrentCovers<T> {
    covers: Vec<Vec<usize>>,
    weights: Vec<f64>,
    item_weights: Vec<f64>,
    est_logtotal: f64,
    cover_iter: T,
    exhausted: bool,
    alpha: f64,
}

impl<T> CurrentCovers<T> {
    pub fn new(
        cover_weights: impl IntoIterator<Item = f64>,
        est_total: f64,
        cover_iter: T,
        coverage: f64,
    ) -> Self {
        CurrentCovers {
            covers: vec![],
            weights: vec![],
            item_weights: cover_weights.into_iter().collect(),
            est_logtotal: est_total,
            cover_iter,
            exhausted: false,
            alpha: coverage.ln(),
        }
    }
    pub fn get_logweight(&self) -> f64 {
        logsumexp(self.weights.iter().map(|&x| -x).collect())
    }
    pub fn est_remain_logweight(&self) -> f64 {
        if self.exhausted {
            return f64::NEG_INFINITY;
        }
        if let Some(min_logweight) = self.weights.last() {
            -min_logweight
                + logdiffexp(
                    self.est_logtotal,
                    (self.weights.len() as f64).ln(),
                )
        } else {
            0.
        }
    }
    pub fn est_logproportion(&self) -> f64 {
        let est_remain = self.est_remain_logweight();
        let cur_cover = self.get_logweight();
        est_remain - logsumexp(vec![est_remain, cur_cover])
    }
}

impl<T: Iterator<Item = impl IntoIterator<Item = usize>>> CurrentCovers<T> {
    pub fn add_new(&mut self) {
        if let Some(new_cover) = self.cover_iter.next() {
            let cover_vec: Vec<usize> = new_cover.into_iter().collect();
            self.weights
                .push(cover_vec.iter().map(|&x| self.item_weights[x]).sum());
            self.covers.push(cover_vec);
        } else {
            self.exhausted = true;
        }
    }
}

impl<T: Iterator<Item = impl IntoIterator<Item = usize>>> Iterator
    for CurrentCovers<T>
{
    type Item = Vec<usize>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.est_logproportion() > self.alpha {
            self.add_new();
            if self.exhausted {
                None
            } else {
                self.covers.last().cloned()
            }
        } else {
            None
        }
    }
}

/// Gets top partitions with coverage `alpha` of the union set of `covers`.
pub fn get_topp_covers(
    covers: Vec<Vec<usize>>,
    weights: Vec<f64>,
    alpha: f64,
    max_iter: Option<usize>,
    max_heap: Option<usize>,
) -> Option<impl IntoIterator<Item = impl IntoIterator<Item = usize>>> {
    if !alpha.is_finite() || alpha.is_sign_negative() || alpha > 1. {
        None
    } else {
        let mut total_covers: HashSet<usize> = HashSet::new();
        for cover in covers.iter().cloned() {
            total_covers.extend(cover);
        }
        let est_total = log_bell_number(total_covers.len() as f64);
        let search_iter =
            get_top_covers(covers, weights.clone(), max_iter, max_heap)
                .into_iter();

        let covers_search =
            CurrentCovers::new(weights, est_total, search_iter, alpha);

        Some(covers_search)
    }
}

#[cfg(test)]
mod tests {
    use super::get_topp_covers;

    /// run example from [Wikipedia][1]
    ///
    /// [1]: https://en.wikipedia.org/wiki/Knuth%27s_Algorithm_X
    #[test]
    fn get_topp_covers_order() {
        let covers: Vec<Vec<usize>> = vec![
            vec![1],
            vec![2],
            vec![1, 2],
            vec![3],
            vec![1, 3],
            vec![2, 3],
            vec![1, 2, 3],
            vec![4],
            vec![1, 4],
            vec![2, 4],
            vec![3, 4],
            vec![1, 2, 4],
            vec![1, 3, 4],
            vec![2, 3, 4],
            vec![1, 2, 3, 4],
            vec![5],
            vec![1, 5],
            vec![2, 5],
            vec![3, 5],
            vec![4, 5],
            vec![1, 2, 5],
            vec![1, 3, 5],
            vec![1, 4, 5],
            vec![2, 3, 5],
            vec![2, 4, 5],
            vec![3, 4, 5],
            vec![1, 2, 3, 5],
            vec![1, 2, 4, 5],
            vec![1, 3, 4, 5],
            vec![2, 3, 4, 5],
            vec![1, 2, 3, 4, 5],
            vec![6],
            vec![1, 6],
            vec![2, 6],
            vec![3, 6],
            vec![4, 6],
            vec![5, 6],
            vec![1, 2, 6],
            vec![1, 3, 6],
            vec![1, 4, 6],
            vec![1, 5, 6],
            vec![2, 3, 6],
            vec![2, 4, 6],
            vec![2, 5, 6],
            vec![3, 4, 6],
            vec![3, 5, 6],
            vec![4, 5, 6],
            vec![1, 2, 3, 6],
            vec![1, 2, 4, 6],
            vec![1, 2, 5, 6],
            vec![1, 3, 4, 6],
            vec![1, 3, 5, 6],
            vec![1, 4, 5, 6],
            vec![2, 3, 4, 6],
            vec![2, 3, 5, 6],
            vec![2, 4, 5, 6],
            vec![3, 4, 5, 6],
            vec![1, 2, 3, 4, 6],
            vec![1, 2, 3, 5, 6],
            vec![1, 2, 4, 5, 6],
            vec![1, 3, 4, 5, 6],
            vec![2, 3, 4, 5, 6],
            vec![1, 2, 3, 4, 5, 6],
            vec![7],
            vec![1, 7],
            vec![2, 7],
            vec![3, 7],
            vec![4, 7],
            vec![5, 7],
            vec![6, 7],
            vec![1, 2, 7],
            vec![1, 3, 7],
            vec![1, 4, 7],
            vec![1, 5, 7],
            vec![1, 6, 7],
            vec![2, 3, 7],
            vec![2, 4, 7],
            vec![2, 5, 7],
            vec![2, 6, 7],
            vec![3, 4, 7],
            vec![3, 5, 7],
            vec![3, 6, 7],
            vec![4, 5, 7],
            vec![4, 6, 7],
            vec![5, 6, 7],
            vec![1, 2, 3, 7],
            vec![1, 2, 4, 7],
            vec![1, 2, 5, 7],
            vec![1, 2, 6, 7],
            vec![1, 3, 4, 7],
            vec![1, 3, 5, 7],
            vec![1, 3, 6, 7],
            vec![1, 4, 5, 7],
            vec![1, 4, 6, 7],
            vec![1, 5, 6, 7],
            vec![2, 3, 4, 7],
            vec![2, 3, 5, 7],
            vec![2, 3, 6, 7],
            vec![2, 4, 5, 7],
            vec![2, 4, 6, 7],
            vec![2, 5, 6, 7],
            vec![3, 4, 5, 7],
            vec![3, 4, 6, 7],
            vec![3, 5, 6, 7],
            vec![4, 5, 6, 7],
            vec![1, 2, 3, 4, 7],
            vec![1, 2, 3, 5, 7],
            vec![1, 2, 3, 6, 7],
            vec![1, 2, 4, 5, 7],
            vec![1, 2, 4, 6, 7],
            vec![1, 2, 5, 6, 7],
            vec![1, 3, 4, 5, 7],
            vec![1, 3, 4, 6, 7],
            vec![1, 3, 5, 6, 7],
            vec![1, 4, 5, 6, 7],
            vec![2, 3, 4, 5, 7],
            vec![2, 3, 4, 6, 7],
            vec![2, 3, 5, 6, 7],
            vec![2, 4, 5, 6, 7],
            vec![3, 4, 5, 6, 7],
            vec![1, 2, 3, 4, 5, 7],
            vec![1, 2, 3, 4, 6, 7],
            vec![1, 2, 3, 5, 6, 7],
            vec![1, 2, 4, 5, 6, 7],
            vec![1, 3, 4, 5, 6, 7],
            vec![2, 3, 4, 5, 6, 7],
            vec![1, 2, 3, 4, 5, 6, 7],
        ];
        let mut weights = vec![1.; covers.len()];
        weights[0] = 10.;
        weights[1] = 10.;
        weights[2] = 10.;
        weights[3] = 10.;
        weights[4] = 10.;
        weights[5] = 10.;
        weights[6] = 10.;
        let desired = 220;

        let results: Vec<Vec<usize>> =
            get_topp_covers(covers, weights, 0.5, None, None)
                .unwrap()
                .into_iter()
                .map(|x| {
                    let mut vn: Vec<usize> = Vec::from_iter(x);
                    vn.sort();
                    vn
                })
                .collect();
        assert_eq!(results.len(), desired);
    }
}
