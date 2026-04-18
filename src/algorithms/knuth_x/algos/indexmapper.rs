use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use itertools::Itertools;

#[derive(Debug)]
pub struct Mapper<T> {
    forwardmap: HashMap<T, usize>,
    reversemap: Vec<T>,
}

impl<T: Clone + Eq + Hash + Ord> Mapper<T> {
    pub fn from_iter(
        iter: impl IntoIterator<Item = impl IntoIterator<Item = T>>,
    ) -> Mapper<T> {
        let mut cmap: HashMap<T, usize> = HashMap::new();
        for row in iter {
            let hs: HashSet<T> = std::collections::HashSet::from_iter(row);
            for item in hs.into_iter() {
                *cmap.entry(item).or_insert(0) += 1;
            }
        }
        let rmap: Vec<T> = cmap
            .into_iter()
            .sorted_by(|x, y| (x.1, &x.0).cmp(&(y.1, &y.0)))
            .map(|x| x.0)
            .collect();
        let fmap: HashMap<T, usize> = HashMap::from_iter(
            rmap.iter().cloned().enumerate().map(|(x, y)| (y, x)),
        );
        Mapper {
            forwardmap: fmap,
            reversemap: rmap,
        }
    }
    pub fn fmap(&self, key: &T) -> Option<usize> {
        self.forwardmap.get(key).copied()
    }
    // pub fn rmap(&self, key: usize) -> Option<&T> {
    //     self.reversemap.get(key)
    // }
    pub fn len(&self) -> usize {
        self.reversemap.len()
    }
}
