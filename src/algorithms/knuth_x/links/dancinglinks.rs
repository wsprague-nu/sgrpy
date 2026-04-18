use itertools::Itertools;

// use super::;
use super::{
    ColumnVec, Cover, HeadAddress, LinkAddress, Node, NodeVec, RowAddress,
};

#[derive(Debug)]
pub enum ColumnSelect {
    Column(usize),
    Fail,
    Complete,
}

/// Dancing links state structure.
#[derive(Debug)]
pub struct DancingLinks {
    /// Vector of column headers.
    columns: ColumnVec,
    /// Vector of nodes.
    nodes: NodeVec,
}

impl DancingLinks {
    /// Initialize from Iterator objects for usize labels.
    pub fn from_iter(
        iter: impl IntoIterator<Item = impl IntoIterator<Item = usize>>,
        num_columns: usize,
    ) -> Self {
        // first, create the column headers
        let head_vec = ColumnVec::new(num_columns);

        // next, create the head nodes
        let node_vec = NodeVec::new(num_columns);

        // with these linked, create the initial DancingLinks structure
        let mut links = DancingLinks {
            columns: head_vec,
            nodes: node_vec,
        };

        for (row, cover) in iter.into_iter().enumerate() {
            links._add_cover(cover, row);
        }

        links
    }
    fn _add_cover(
        &mut self,
        cover: impl IntoIterator<Item = usize>,
        row: usize,
    ) {
        // condense a cover object
        let cover = Cover::from_iter(cover);

        // vector of column header indices for cover
        let new_headers: Vec<_> =
            cover.into_iter().map(|x| HeadAddress(x + 1)).collect();

        // number of new nodes
        let num_new = new_headers.len();

        // vector of new node indices
        let new_nodes: Vec<_> = (self.num_nodes()
            ..(self.num_nodes() + num_new))
            .map(LinkAddress)
            .collect();

        // add new nodes
        for (cur_col, &cur_node) in std::iter::zip(new_headers, &new_nodes) {
            let head = self.columns[cur_col].node;
            let head_up = self.nodes[head].up;

            // temporarily use current node address for left/right
            let new_node = Node {
                col: cur_col,
                row: Some(RowAddress(row)),
                up: head_up,
                down: head,
                left: cur_node,
                right: cur_node,
            };
            self.nodes[head].up = cur_node;
            self.nodes[head_up].down = cur_node;
            self.columns[cur_col].size += 1;
            self.nodes.push(new_node);
        }

        for (&left, &right) in new_nodes.iter().circular_tuple_windows() {
            self.nodes[left].right = right;
            self.nodes[right].left = left;
        }
    }

    /// Cover column at index `col`
    pub fn cover_column(&mut self, col: usize) {
        self._cover_column(HeadAddress(col + 1));
    }

    fn _cover_column(&mut self, col_i: HeadAddress<usize>) {
        let col_node = self.columns[col_i].node;
        let col_node_right = self.nodes[col_node].right;
        let col_node_left = self.nodes[col_node].left;
        self.nodes[col_node_right].left = col_node_left;
        self.nodes[col_node_left].right = col_node_right;
        let mut i = self.nodes[col_node].down;
        while i != col_node {
            let mut j = self.nodes[i].right;
            while j != i {
                let j_down = self.nodes[j].down;
                let j_up = self.nodes[j].up;
                self.nodes[j_down].up = j_up;
                self.nodes[j_up].down = j_down;
                self.columns[self.nodes[j].col].size -= 1;
                j = self.nodes[j].right;
            }
            i = self.nodes[i].down;
        }
    }

    /// Uncover column at index `col`
    pub fn uncover_column(&mut self, col: usize) {
        self._uncover_column(HeadAddress(col + 1));
    }
    fn _uncover_column(&mut self, col_i: HeadAddress<usize>) {
        let col_node = self.columns[col_i].node;
        let mut i = self.nodes[col_node].up;
        while i != col_node {
            let mut j = self.nodes[i].left;
            while j != i {
                let j_down = self.nodes[j].down;
                let j_up = self.nodes[j].up;
                self.columns[self.nodes[j].col].size += 1;
                self.nodes[j_down].up = j;
                self.nodes[j_up].down = j;
                j = self.nodes[j].left;
            }
            i = self.nodes[i].up;
        }
        let col_node_right = self.nodes[col_node].right;
        let col_node_left = self.nodes[col_node].left;
        self.nodes[col_node_right].left = col_node;
        self.nodes[col_node_left].right = col_node;
    }

    pub fn get_rows(&self, col: usize) -> Vec<usize> {
        let head = LinkAddress(col);
        let mut i = self.nodes[head].down;
        let mut row_vec = Vec::new();
        while i != head {
            if let Some(row) = self.nodes[i].row {
                row_vec.push(row.0);
            }
            i = self.nodes[i].down;
        }
        row_vec
    }

    // pub fn get_active_cols(&self) -> Vec<usize> {
    //     let head = LinkAddress(0);
    //     let mut cur_head = self.nodes[head].right;
    //     let mut active_cols: Vec<usize> = Vec::new();
    //     while cur_head != head {
    //         active_cols.push(self.nodes[cur_head].col.0);
    //         cur_head = self.nodes[cur_head].right;
    //     }
    //     active_cols
    // }

    pub fn select_column(&self) -> ColumnSelect {
        let head_node = LinkAddress(0);
        let mut cur_head = self.nodes[head_node].right;
        let mut best_val = None;
        let mut best_col = None;
        while cur_head != head_node {
            let cur_col = self.nodes[cur_head].col;
            let cur_val = self.columns[cur_col].size;
            if cur_val == 0 {
                return ColumnSelect::Fail;
            }
            match best_val {
                None => {
                    best_val = Some(cur_val);
                    best_col = Some(cur_col);
                }
                Some(best_val_unw) => {
                    if cur_val < best_val_unw {
                        best_val = Some(cur_val);
                        best_col = Some(cur_col);
                    }
                }
            }
            cur_head = self.nodes[cur_head].right;
        }
        match best_col {
            None => ColumnSelect::Complete,
            Some(col_value) => ColumnSelect::Column(col_value.0),
        }
    }

    // pub fn num_cols(&self) -> usize {
    //     self.columns.len()
    // }
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::DancingLinks;

    #[test]
    fn create() {
        let covers: Vec<Vec<usize>> = vec![
            vec![0, 3, 6],
            vec![0, 3],
            vec![3, 4, 6],
            vec![2, 4, 5],
            vec![1, 2, 5, 6],
            vec![1, 6],
        ];
        let mut links: DancingLinks =
            DancingLinks::from_iter(covers.clone(), 7);

        // assert_eq!(
        //     links.columns.len(),
        //     8,
        //     "Test length {} equals {}",
        //     links.columns.len(),
        //     covers.len()
        // );

        links.cover_column(0);
        links.uncover_column(0);
    }
}
