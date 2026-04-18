use super::{HeadAddress, LinkAddress, RowAddress};

/// Node information.
#[derive(Debug)]
pub struct Node<N, C, R> {
    /// Index of column header node.
    pub col: HeadAddress<C>,
    /// Index of row.
    pub row: Option<RowAddress<R>>,
    /// Index of node "above" this one (wrapping).
    pub up: LinkAddress<N>,
    /// Index of node "below" this one (wrapping).
    pub down: LinkAddress<N>,
    /// Index of node "left" of this one (wrapping).
    pub left: LinkAddress<N>,
    /// Index of node "right" of this one (wrapping).
    pub right: LinkAddress<N>,
}
