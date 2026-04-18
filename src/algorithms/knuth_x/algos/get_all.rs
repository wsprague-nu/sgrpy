use std::{collections::HashSet, hash::Hash};

use super::Mapper;
use super::search::{LeafNode, NodeStatus, SearchResult, SearchState};

/// Gets all partitions of the union set of `iter`.
pub fn get_all_covers<T: Clone + Eq + Hash + Ord>(
    iter: impl IntoIterator<Item = impl IntoIterator<Item = T>>,
) -> impl IntoIterator<Item = impl IntoIterator<Item = usize>> {
    let search_iter = get_all_states(iter);

    let mut result_vec = Vec::new();
    for result in search_iter {
        if let SearchResult(total_cover, NodeStatus::Complete) = result {
            result_vec.push(total_cover.cov);
        }
    }
    result_vec
}

pub fn get_all_states<T: Clone + Eq + Hash + Ord>(
    iter: impl IntoIterator<Item = impl IntoIterator<Item = T>>,
) -> impl IntoIterator<Item = SearchResult> {
    let iter_vec: Vec<Vec<T>> =
        iter.into_iter().map(|x| x.into_iter().collect()).collect();
    let mapper: Mapper<T> = Mapper::from_iter(iter_vec.iter().cloned());
    let iter_true = iter_vec.into_iter().map(|row| {
        HashSet::<usize>::from_iter(
            row.into_iter().map(|item| mapper.fmap(&item).unwrap()),
        )
    });
    SearchState::from_iter(
        iter_true,
        mapper.len(),
        vec![LeafNode::new()],
        |links| links.select_column(),
        None,
    )
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use super::{NodeStatus, get_all_covers, get_all_states};

    /// run example from [Wikipedia][1]
    ///
    /// [1]: https://en.wikipedia.org/wiki/Knuth%27s_Algorithm_X
    #[test]
    fn wikipedia_covers() {
        let covers: Vec<Vec<usize>> = vec![
            vec![1, 4, 7],
            vec![1, 4],
            vec![4, 5, 7],
            vec![3, 5, 6],
            vec![2, 3, 6, 7],
            vec![2, 7],
        ];
        assert_eq!(
            get_all_covers(covers)
                .into_iter()
                .map(|x| {
                    let mut vn = Vec::from_iter(x);
                    vn.sort();
                    vn
                })
                .sorted()
                .collect_vec(),
            vec![vec![1, 3, 5]]
        );
    }

    #[test]
    /// run example from [Wikipedia][1]
    ///
    /// [1]: https://en.wikipedia.org/wiki/Knuth%27s_Algorithm_X
    fn wikipedia_states() {
        let covers: Vec<Vec<usize>> = vec![
            vec![1, 4, 7],
            vec![1, 4],
            vec![4, 5, 7],
            vec![3, 5, 6],
            vec![2, 3, 6, 7],
            vec![2, 7],
        ];
        let target = vec![
            (vec![], NodeStatus::Partial),
            (vec![0], NodeStatus::Fail),
            (vec![1], NodeStatus::Partial),
            (vec![1, 3], NodeStatus::Partial),
            (vec![1, 3, 5], NodeStatus::Complete),
        ]
        .into_iter()
        .sorted()
        .collect_vec();
        let result = get_all_states(covers)
            .into_iter()
            .map(|x| {
                let mut vn = Vec::from_iter(x.0.cov);
                vn.sort();
                (vn, x.1)
            })
            .sorted()
            .collect_vec();
        assert_eq!(result, target);
    }
}
