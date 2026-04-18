//! Algorithms useful for the implementation of the SGR method.

#![deny(deprecated_safe, future_incompatible, rust_2024_compatibility)]
#![warn(missing_debug_implementations, missing_docs, nonstandard_style, unused)]

mod algorithms;

#[pyo3::pymodule(name = "dl_rs")]
mod dl_rs {
    use super::algorithms::knuth_x::get_all as get_all_rs;
    use super::algorithms::knuth_x::get_top as get_top_rs;
    use super::algorithms::knuth_x::get_topn as get_topn_rs;
    use super::algorithms::knuth_x::get_topn_sc as get_topn_sc_rs;
    use super::algorithms::knuth_x::get_topp as get_topp_rs;

    /// Get all partitions given a set of covers
    #[pyo3::pyfunction]
    fn get_all(partitions: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
        get_all_rs(partitions)
            .into_iter()
            .map(Vec::from_iter)
            .collect()
    }

    /// Get partitions in order of decreasing likelihood, given partition weights
    #[pyo3::pyfunction]
    #[pyo3(signature = (partitions, weights, max_iter=None, max_heap=None))]
    fn get_top(
        partitions: Vec<Vec<usize>>,
        weights: Vec<f64>,
        max_iter: Option<usize>,
        max_heap: Option<usize>,
    ) -> Vec<Vec<usize>> {
        get_top_rs(partitions, weights, max_iter, max_heap)
            .into_iter()
            .map(Vec::from_iter)
            .collect()
    }

    /// Get top `n` partitions in order of decreasing likelihood, given partition weights
    #[pyo3::pyfunction]
    #[pyo3(signature = (partitions, weights, n, max_iter=None, max_heap=None))]
    fn get_topn(
        partitions: Vec<Vec<usize>>,
        weights: Vec<f64>,
        n: usize,
        max_iter: Option<usize>,
        max_heap: Option<usize>,
    ) -> Vec<Vec<usize>> {
        get_topn_rs(partitions, weights, n, max_iter, max_heap)
            .into_iter()
            .map(Vec::from_iter)
            .collect()
    }

    /// Get top `n` partitions in order of decreasing likelihood, given partition weights
    #[pyo3::pyfunction]
    #[pyo3(signature = (partitions, weights, edge_counts, edge_total, labels, vertex_total, n, max_iter=None, max_heap=None))]
    fn get_topn_sc(
        partitions: Vec<Vec<usize>>,
        weights: Vec<f64>,
        edge_counts: Vec<usize>,
        edge_total: usize,
        labels: Vec<usize>,
        vertex_total: usize,
        n: usize,
        max_iter: Option<usize>,
        max_heap: Option<usize>,
    ) -> Vec<Vec<usize>> {
        get_topn_sc_rs(
            partitions,
            weights,
            edge_counts,
            edge_total,
            labels,
            vertex_total,
            n,
            max_iter,
            max_heap,
        )
        .into_iter()
        .map(Vec::from_iter)
        .collect()
    }

    /// Get top partitions in order of decreasing likelihood, with total
    /// probability coverage greater than `1 - alpha`
    #[pyo3::pyfunction]
    #[pyo3(signature = (partitions, weights, alpha, max_iter=None, max_heap=None))]
    fn get_topp(
        partitions: Vec<Vec<usize>>,
        weights: Vec<f64>,
        alpha: f64,
        max_iter: Option<usize>,
        max_heap: Option<usize>,
    ) -> Vec<Vec<usize>> {
        if let Some(result) =
            get_topp_rs(partitions, weights, alpha, max_iter, max_heap)
        {
            result.into_iter().map(Vec::from_iter).collect()
        } else {
            vec![]
        }
    }
}

// #[pyo3::pymodule(name = "dl_rs")]
// mod python_interface {
//     use super::algorithms::knuth_x::get_all as get_all_rs;

//     /// Get all partitions given a set of covers
//     #[pyo3::pyfunction]
//     fn get_all(partitions: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
//         get_all_rs(partitions)
//             .into_iter()
//             .map(Vec::from_iter)
//             .collect()
//     }
// }
