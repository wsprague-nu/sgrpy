use super::linkstate::RowSelect;

use super::{ColumnSelect, DancingLinks, HeapTrait, LinkState, MoveList};

/// Leaf node containing used covers and current moves
#[derive(Debug)]
pub struct LeafNode {
    /// Array of covers
    pub cov: Vec<usize>,

    /// Moves list
    pub pos: MoveList,
}

impl LeafNode {
    pub fn new() -> LeafNode {
        LeafNode {
            cov: Vec::new(),
            pos: MoveList(Vec::new()),
        }
    }
}

#[derive(Debug, Ord, PartialOrd, PartialEq, Eq)]
pub enum NodeStatus {
    Partial,
    Complete,
    Fail,
}

#[derive(Debug)]
pub struct SearchResult(pub LeafNode, pub NodeStatus);

// impl<T> Ord for LeafNode<T> {
//     fn cm
// }

#[derive(Debug)]
pub struct SearchState<H, F> {
    heap: H,
    links: LinkState<F>,
    covers: Vec<Vec<usize>>,
    max_iter: Option<usize>,
}

impl<H: HeapTrait<LeafNode>, F> SearchState<H, F>
where
    F: Fn(&DancingLinks) -> ColumnSelect,
{
    /// Initialize search state from iterator
    pub fn from_iter(
        iter: impl IntoIterator<Item = impl IntoIterator<Item = usize>>,
        num_columns: usize,
        heap: H,
        select_func: F,
        max_iter: Option<usize>,
    ) -> SearchState<H, F> {
        let vecs: Vec<_> =
            iter.into_iter().map(|x| x.into_iter().collect()).collect();
        let links = LinkState::from_iter(
            vecs.iter().cloned(),
            num_columns,
            select_func,
        );
        // println!("SearchState Initialized");
        SearchState {
            heap,
            links,
            covers: vecs,
            max_iter,
        }
    }
}

impl<H: HeapTrait<LeafNode>, F> Iterator for SearchState<H, F>
where
    F: Fn(&DancingLinks) -> ColumnSelect,
{
    type Item = SearchResult;

    /// Find and return next search state
    fn next(&mut self) -> Option<Self::Item> {
        match self.max_iter {
            None => {}
            Some(0) => return None,
            Some(ref mut x) => *x -= 1,
        }
        // pop current leaf, if any
        if let Some(cur_leaf) = self.heap.pop() {
            // move to leaf
            self.links.move_to(cur_leaf.pos.clone());

            // get options from here
            let options = self.links.get_options();
            match options {
                RowSelect::Complete => {
                    // println!("Current heap size: {:}", self.heap.len());
                    // println!("Completed partition");
                    Some(SearchResult(cur_leaf, NodeStatus::Complete))
                }
                RowSelect::Fail => {
                    // println!("Current heap size: {:}", self.heap.len());
                    Some(SearchResult(cur_leaf, NodeStatus::Fail))
                }
                RowSelect::Rows(rows) => {
                    for row in rows.into_iter().rev() {
                        let mut cov = cur_leaf.cov.clone();
                        cov.push(row);
                        let mut pos = cur_leaf.pos.clone();
                        pos.0.extend(self.covers[row].iter());
                        let new_leaf = LeafNode { cov, pos };
                        self.heap.push(new_leaf);
                    }
                    // println!("Current heap size: {:}", self.heap.len());
                    Some(SearchResult(cur_leaf, NodeStatus::Partial))
                }
            }
        } else {
            None
        }
    }
}
