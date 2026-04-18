use std::iter::{IntoIterator, Iterator};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Cover<N> {
    covers: Vec<N>,
}

impl<N: Ord> Cover<N> {
    pub fn new(cover: Vec<N>) -> Cover<N> {
        Cover::from_iter(cover)
    }
    pub fn from_iter(cover: impl IntoIterator<Item = N>) -> Cover<N> {
        let bset = std::collections::BTreeSet::from_iter(cover)
            .into_iter()
            .collect();
        Cover { covers: bset }
    }
}

impl<N> Cover<N> {
    pub fn get_covers(&self) -> &Vec<N> {
        &self.covers
    }
}

impl<'a, N: Copy> IntoIterator for &'a Cover<N> {
    type Item = N;
    type IntoIter = CoverIterator<'a, N>;
    fn into_iter(self) -> Self::IntoIter {
        CoverIterator {
            cover: self,
            index: 0,
        }
    }
}

#[derive(Debug)]
pub struct CoverIterator<'a, N> {
    cover: &'a Cover<N>,
    index: usize,
}

impl<N: Copy> Iterator for CoverIterator<'_, N> {
    type Item = N;
    fn next(&mut self) -> Option<N> {
        if self.index >= self.cover.covers.len() {
            return None;
        }
        let result = self.cover.covers[self.index];
        self.index += 1;
        Some(result)
    }
}
