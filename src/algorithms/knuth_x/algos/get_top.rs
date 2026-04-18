use std::{collections::HashSet, hash::Hash};

use itertools::Itertools;
use ordered_float::OrderedFloat;

use super::super::links::{ColumnSelect, DancingLinks};
use super::Mapper;
use super::search::{
    LeafNode, LimFuncHeap, NodeStatus, SearchResult, SearchState,
};
use super::utils::{calc_ac_factor, calc_as_factor};

/// Gets partitions of the union set of `covers` in order of lowest weight.
pub fn get_top_covers(
    covers: Vec<Vec<usize>>,
    weights: Vec<f64>,
    max_iter: Option<usize>,
    max_heap: Option<usize>,
) -> impl IntoIterator<Item = impl IntoIterator<Item = usize>> {
    let mut item_set = std::collections::HashSet::new();
    for vec in covers.iter() {
        item_set.extend(vec.iter().cloned());
    }
    // let select_finder =
    //     super::col_select::ColSelector::from_costs(weights.clone());
    let min_estimator =
        super::MinEstAvg::from_examples(covers.iter().cloned().zip(weights));
    let rating_func = move |x: &Vec<usize>| {
        let (cur_cost, fut_cost) = min_estimator.min_value(x);
        // println!("Cover: {:?}", x);
        // println!("Cost: {:?}", result);
        // println!("Future Cost: {:?}", future_cost);
        // println!("Total Cost: {:?}", result + future_cost);
        // println!();
        // pause();
        cur_cost + fut_cost
    };
    // This could be changed back
    let select_func = |d: &DancingLinks| d.select_column();
    // let select_func = move |d: &DancingLinks| select_finder.select_col(d);
    let search_iter =
        get_top_states(covers, rating_func, select_func, max_iter, max_heap);

    search_iter.into_iter().filter_map(|sr| {
        if let SearchResult(total_cover, NodeStatus::Complete) = sr {
            Some(total_cover.cov)
        } else {
            None
        }
    })
}

/// Gets partitions of the union set of `covers` in order of lowest weight.
///
/// Uses math relevant to number of vertex counts.
pub fn get_top_covers_sc<T: Clone + Eq + Hash + Ord>(
    covers: Vec<Vec<T>>,
    weights: Vec<f64>,
    edge_counts: Vec<usize>,
    edge_total: usize,
    labels: Vec<usize>,
    vertex_total: usize,
    max_iter: Option<usize>,
    max_heap: Option<usize>,
) -> impl IntoIterator<Item = impl IntoIterator<Item = usize>> {
    let vertex_counts = Vec::from_iter(covers.iter().map(|x| x.len()));
    let mut item_set = std::collections::HashSet::new();
    for vec in covers.iter() {
        item_set.extend(vec.iter().cloned());
    }
    let total_size = item_set.len();
    let min_estimator = super::min_est::PartMin::from_examples(
        vertex_counts.clone(),
        weights.clone(),
        total_size,
    );
    let rating_func = move |x: &Vec<usize>| {
        let vertex_vec = Vec::from_iter(x.iter().map(|&y| vertex_counts[y]));
        let cur_coverage = vertex_vec.iter().sum::<usize>();
        let label_vec = x
            .iter()
            .map(|&v| labels[v])
            .counts()
            .into_values()
            .collect_vec();
        let remaining = total_size - cur_coverage;
        let future_cost = min_estimator.min_value(remaining);

        if cur_coverage >= vertex_total {
            x.iter().map(|&y| weights[y]).sum::<f64>()
                + calc_ac_factor(label_vec, 0)
                + calc_as_factor(x.iter().map(|&y| edge_counts[y]), edge_total)
        } else {
            x.iter().map(|&y| weights[y]).sum::<f64>()
                + calc_ac_factor(label_vec, vertex_total - cur_coverage)
                + future_cost
        }
    };
    let search_iter = get_top_states(
        covers,
        rating_func,
        |links| links.select_column(),
        max_iter,
        max_heap,
    );

    search_iter.into_iter().filter_map(|sr| {
        if let SearchResult(total_cover, NodeStatus::Complete) = sr {
            Some(total_cover.cov)
        } else {
            None
        }
    })
}

pub fn get_top_states<T: Clone + Eq + Hash + Ord>(
    covers: Vec<Vec<T>>,
    rating_func: impl Fn(&Vec<usize>) -> f64,
    select_func: impl Fn(&DancingLinks) -> ColumnSelect,
    max_iter: Option<usize>,
    max_heap: Option<usize>,
) -> impl IntoIterator<Item = SearchResult> {
    let iter_vec: Vec<Vec<T>> = covers
        .into_iter()
        .map(|x| x.into_iter().collect())
        .collect();
    let mapper: Mapper<T> = Mapper::from_iter(iter_vec.iter().cloned());
    let iter_true = iter_vec.into_iter().map(|row| {
        HashSet::<usize>::from_iter(
            row.into_iter().map(|item| mapper.fmap(&item).unwrap()),
        )
    });
    let heap = LimFuncHeap::new_single(
        LeafNode::new(),
        move |x: &LeafNode| OrderedFloat(rating_func(&x.cov)),
        max_heap,
    );
    SearchState::from_iter(iter_true, mapper.len(), heap, select_func, max_iter)
}

#[cfg(test)]
mod tests {
    use super::{get_top_covers, get_top_covers_sc};

    /// run example from [Wikipedia][1]
    ///
    /// [1]: https://en.wikipedia.org/wiki/Knuth%27s_Algorithm_X
    #[test]
    fn get_top_covers_order() {
        let covers: Vec<Vec<usize>> = vec![
            vec![1],
            vec![2],
            vec![3],
            vec![4],
            vec![5],
            vec![6],
            vec![7],
            vec![1, 2],
            vec![1, 2, 3],
            vec![3, 4],
        ];
        let weights = vec![1.; 10];
        let desired = vec![
            vec![4, 5, 6, 7, 9],
            vec![3, 4, 5, 6, 8],
            vec![0, 1, 4, 5, 6, 9],
            vec![2, 3, 4, 5, 6, 7],
            vec![0, 1, 2, 3, 4, 5, 6],
        ];

        let results: Vec<Vec<usize>> =
            get_top_covers(covers, weights, None, None)
                .into_iter()
                .map(|x| {
                    let mut vn = Vec::from_iter(x);
                    vn.sort();
                    vn
                })
                .collect();
        assert_eq!(results, desired);
    }
    #[test]
    fn get_top_covers_sc_order() {
        let covers: Vec<Vec<usize>> = vec![
            vec![1],
            vec![2],
            vec![3],
            vec![4],
            vec![5],
            vec![6],
            vec![7],
            vec![1, 2],
            vec![2, 3],
            vec![3, 4],
            vec![4, 5],
            vec![5, 6],
            vec![6, 1],
            vec![6, 7],
            vec![1, 2, 3],
            vec![2, 3, 4],
            vec![3, 4, 5],
            vec![4, 5, 6],
            vec![5, 6, 1],
            vec![6, 1, 2],
            vec![5, 6, 7],
            vec![7, 6, 1],
            vec![1, 2, 3, 4],
            vec![2, 3, 4, 5],
            vec![3, 4, 5, 6],
            vec![4, 5, 6, 1],
            vec![4, 5, 6, 7],
            vec![5, 6, 1, 2],
            vec![5, 6, 1, 7],
            vec![2, 1, 6, 7],
            vec![1, 2, 3, 4, 5],
            vec![2, 3, 4, 5, 6],
            vec![3, 4, 5, 6, 1],
            vec![3, 4, 5, 6, 7],
            vec![4, 5, 6, 1, 2],
            vec![4, 5, 6, 1, 7],
            vec![5, 6, 1, 2, 3],
            vec![5, 6, 1, 2, 7],
            vec![6, 1, 2, 3, 4],
            vec![6, 1, 2, 3, 7],
            vec![1, 2, 3, 4, 5, 6],
            vec![2, 3, 4, 5, 6, 7],
            vec![1, 3, 4, 5, 6, 7],
            vec![1, 2, 4, 5, 6, 7],
            vec![1, 2, 3, 5, 6, 7],
            vec![1, 2, 3, 4, 6, 7],
            vec![1, 2, 3, 4, 5, 6, 7],
        ];
        let weights = vec![17f64.ln(); covers.len()];
        let labels = vec![
            1, 1, 1, 1, 1, 1, 2, 3, 3, 3, 3, 3, 3, 4, 5, 5, 5, 5, 5, 5, 6, 6,
            7, 7, 7, 7, 8, 7, 9, 8, 10, 10, 10, 11, 10, 12, 10, 12, 10, 11, 13,
            14, 15, 16, 15, 14, 17,
        ];
        let ecounts: Vec<usize> = vec![
            0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
            3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 5, 5, 5,
            5, 5, 7,
        ];
        let desired = vec![
            vec![46],
            vec![6, 40],
            vec![13, 30],
            vec![21, 23],
            vec![20, 22],
            vec![16, 29],
            vec![15, 28],
            vec![14, 26],
            vec![10, 39],
            vec![9, 37],
            vec![8, 35],
            vec![7, 33],
            vec![4, 45],
            vec![3, 44],
            vec![2, 43],
            vec![1, 42],
            vec![0, 41],
            vec![2, 6, 34],
            vec![6, 9, 27],
            vec![6, 8, 25],
            vec![0, 13, 23],
            vec![7, 13, 16],
            vec![1, 16, 21],
            vec![3, 14, 20],
            vec![6, 7, 24],
            vec![6, 12, 23],
            vec![6, 11, 22],
            vec![10, 13, 14],
            vec![2, 10, 29],
            vec![1, 9, 28],
            vec![2, 7, 26],
            vec![4, 6, 38],
            vec![3, 6, 36],
            vec![1, 6, 32],
            vec![0, 6, 31],
            vec![5, 6, 30],
            vec![4, 13, 22],
            vec![4, 15, 21],
            vec![0, 15, 20],
            vec![4, 9, 29],
            vec![3, 8, 28],
            vec![0, 8, 26],
            vec![6, 16, 19],
            vec![6, 15, 18],
            vec![6, 14, 17],
            vec![8, 10, 21],
            vec![7, 9, 20],
            vec![3, 4, 39],
            vec![2, 3, 37],
            vec![1, 2, 35],
            vec![0, 1, 33],
            vec![2, 6, 7, 17],
            vec![1, 6, 9, 18],
            vec![3, 6, 8, 18],
            vec![2, 6, 10, 19],
            vec![1, 6, 12, 16],
            vec![0, 6, 11, 15],
            vec![3, 6, 11, 14],
            vec![4, 6, 9, 19],
            vec![0, 6, 8, 17],
            vec![5, 6, 7, 16],
            vec![4, 6, 12, 15],
            vec![5, 6, 10, 14],
            vec![1, 2, 6, 25],
            vec![0, 1, 13, 16],
            vec![0, 8, 10, 13],
            vec![2, 7, 10, 13],
            vec![1, 2, 10, 21],
            vec![2, 3, 7, 20],
            vec![2, 3, 6, 27],
            vec![0, 4, 13, 15],
            vec![1, 4, 9, 21],
            vec![0, 3, 8, 20],
            vec![0, 1, 6, 24],
            vec![0, 5, 6, 23],
            vec![4, 5, 6, 22],
            vec![3, 4, 13, 14],
            vec![3, 4, 8, 21],
            vec![0, 1, 9, 20],
            vec![4, 7, 9, 13],
            vec![6, 7, 9, 11],
            vec![6, 8, 10, 12],
            vec![2, 3, 4, 29],
            vec![1, 2, 3, 28],
            vec![0, 1, 2, 26],
            vec![1, 2, 6, 10, 12],
            vec![2, 3, 6, 7, 11],
            vec![0, 1, 6, 9, 11],
            vec![0, 3, 6, 8, 11],
            vec![2, 5, 6, 7, 10],
            vec![1, 4, 6, 9, 12],
            vec![3, 4, 6, 8, 12],
            vec![4, 5, 6, 7, 9],
            vec![0, 5, 6, 8, 10],
            vec![1, 2, 3, 6, 18],
            vec![0, 1, 2, 10, 13],
            vec![0, 1, 2, 6, 17],
            vec![0, 1, 4, 9, 13],
            vec![2, 3, 4, 6, 19],
            vec![0, 3, 4, 8, 13],
            vec![0, 1, 5, 6, 16],
            vec![0, 4, 5, 6, 15],
            vec![3, 4, 5, 6, 14],
            vec![2, 3, 4, 7, 13],
            vec![1, 2, 3, 4, 21],
            vec![0, 1, 2, 3, 20],
            vec![0, 1, 2, 3, 6, 11],
            vec![1, 2, 3, 4, 6, 12],
            vec![0, 1, 2, 5, 6, 10],
            vec![2, 3, 4, 5, 6, 7],
            vec![0, 1, 4, 5, 6, 9],
            vec![0, 3, 4, 5, 6, 8],
            vec![0, 1, 2, 3, 4, 13],
            vec![0, 1, 2, 3, 4, 5, 6],
        ];

        let results: Vec<Vec<usize>> = get_top_covers_sc(
            covers, weights, ecounts, 7, labels, 7, None, None,
        )
        .into_iter()
        .map(|x| {
            let mut vn = Vec::from_iter(x);
            vn.sort();
            vn
        })
        .collect();
        assert_eq!(results, desired);
    }
}
