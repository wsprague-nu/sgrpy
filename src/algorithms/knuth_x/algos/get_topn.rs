//! Algorithms which give only top N covers, based on various algorithms.

use super::get_top::{get_top_covers, get_top_covers_sc};

use std::hash::Hash;

/// Gets top `n` partitions of the union set of `covers`.
pub fn get_topn_covers(
    covers: Vec<Vec<usize>>,
    weights: Vec<f64>,
    n: usize,
    max_iter: Option<usize>,
    max_heap: Option<usize>,
) -> impl IntoIterator<Item = impl IntoIterator<Item = usize>> {
    get_top_covers(covers, weights, max_iter, max_heap)
        .into_iter()
        .take(n)
}

/// Gets top `n` partitions of the union set of `covers`, including SC factors.
pub fn get_topn_covers_sc<T: Clone + Eq + Hash + Ord>(
    covers: Vec<Vec<T>>,
    weights: Vec<f64>,
    edge_counts: Vec<usize>,
    edge_total: usize,
    labels: Vec<usize>,
    vertex_total: usize,
    n: usize,
    max_iter: Option<usize>,
    max_heap: Option<usize>,
) -> impl IntoIterator<Item = impl IntoIterator<Item = usize>> {
    get_top_covers_sc(
        covers,
        weights,
        edge_counts,
        edge_total,
        labels,
        vertex_total,
        max_iter,
        max_heap,
    )
    .into_iter()
    .take(n)
}

#[cfg(test)]
mod tests {
    use super::{get_topn_covers, get_topn_covers_sc};

    /// run example from [Wikipedia][1]
    ///
    /// [1]: https://en.wikipedia.org/wiki/Knuth%27s_Algorithm_X
    #[test]
    fn get_topn_covers_order() {
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
        ];

        let results: Vec<Vec<usize>> =
            get_topn_covers(covers, weights, 3, None, None)
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
    fn get_topn_covers_sc_order() {
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
        let labels = vec![
            1, 1, 1, 1, 1, 1, 2, 3, 3, 3, 3, 3, 3, 4, 5, 5, 5, 5, 5, 5, 6, 6,
            7, 7, 7, 7, 8, 7, 9, 8, 10, 10, 10, 11, 10, 12, 10, 12, 10, 11, 13,
            14, 15, 16, 15, 14, 17,
        ];
        let weights = vec![17f64.ln(); covers.len()];
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
        ];

        let results: Vec<Vec<usize>> = get_topn_covers_sc(
            covers, weights, ecounts, 7, labels, 7, 8, None, None,
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
