use std::ops::{Index, IndexMut};

use super::{HeadAddress, LinkAddress, Node};

fn wrap_left(i: usize, shift: usize, upper: usize) -> usize {
    if shift > i {
        upper - shift + i
    } else {
        i - shift
    }
}

fn wrap_right(i: usize, shift: usize, upper: usize) -> usize {
    (i + shift) % upper
}

/// Node vector.
#[derive(Debug)]
pub struct NodeVec {
    vec: Vec<Node<usize, usize, usize>>,
}
impl Index<LinkAddress<usize>> for NodeVec {
    type Output = Node<usize, usize, usize>;
    fn index(&self, index: LinkAddress<usize>) -> &Self::Output {
        &self.vec[index.0]
    }
}
impl IndexMut<LinkAddress<usize>> for NodeVec {
    fn index_mut(&mut self, index: LinkAddress<usize>) -> &mut Self::Output {
        &mut self.vec[index.0]
    }
}
impl NodeVec {
    /// Initialize from column header iterator.
    pub fn from_iter(
        iter: impl IntoIterator<Item = Node<usize, usize, usize>>,
    ) -> Self {
        NodeVec {
            vec: iter.into_iter().collect(),
        }
    }
    /// Initialize from range.
    pub fn new(num_cols: usize) -> Self {
        NodeVec::from_iter((0..=num_cols).map(|x| Node {
            col: HeadAddress(x),
            row: None,
            up: LinkAddress(x),
            down: LinkAddress(x),
            left: LinkAddress(wrap_left(x, 1, num_cols + 1)),
            right: LinkAddress(wrap_right(x, 1, num_cols + 1)),
        }))
    }
    /// Get size of vector.
    pub fn len(&self) -> usize {
        self.vec.len()
    }
    /// Push new node into vector
    pub fn push(&mut self, item: Node<usize, usize, usize>) {
        self.vec.push(item);
    }
}
