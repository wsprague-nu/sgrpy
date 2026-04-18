use std::ops::{Index, IndexMut};

use super::{ColumnHeader, HeadAddress, LinkAddress};

/// Column vector.
#[derive(Debug)]
pub struct ColumnVec {
    /// Internal vector.
    vec: Vec<ColumnHeader<usize>>,
}
impl Index<HeadAddress<usize>> for ColumnVec {
    type Output = ColumnHeader<usize>;
    fn index(&self, index: HeadAddress<usize>) -> &Self::Output {
        &self.vec[index.0]
    }
}
impl IndexMut<HeadAddress<usize>> for ColumnVec {
    fn index_mut(&mut self, index: HeadAddress<usize>) -> &mut Self::Output {
        &mut self.vec[index.0]
    }
}
impl ColumnVec {
    /// Initialize from column header iterator.
    pub fn from_iter(
        iter: impl IntoIterator<Item = ColumnHeader<usize>>,
    ) -> Self {
        ColumnVec {
            vec: iter.into_iter().collect(),
        }
    }
    /// Initialize from range.
    pub fn new(num_cols: usize) -> Self {
        ColumnVec::from_iter((0..=num_cols).map(|x| ColumnHeader {
            // col: if x == 0 { None } else { Some(x - 1) },
            size: 0,
            node: LinkAddress(x),
        }))
    }
    // /// Get size of vector.
    // pub fn len(&self) -> usize {
    //     self.vec.len()
    // }
}
