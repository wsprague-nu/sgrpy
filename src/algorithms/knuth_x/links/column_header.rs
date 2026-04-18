use super::link_address::LinkAddress;

/// Column header information.
#[derive(Debug)]
pub struct ColumnHeader<N> {
    /// Column label.
    ///
    // /// Value of `None` indicates root column.
    // pub col: Option<H>,
    /// Number of nodes in this column.
    pub size: usize,
    /// Index of column header node.
    pub node: LinkAddress<N>,
}
