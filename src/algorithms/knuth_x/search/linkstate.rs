use super::{ColumnSelect, DancingLinks};

use super::MoveList;

#[derive(Debug)]
pub enum RowSelect {
    Rows(Vec<usize>),
    Fail,
    Complete,
}

/// State manager for DancingLinks
#[derive(Debug)]
pub struct LinkState<F> {
    cur_pos: MoveList,
    links: DancingLinks,
    col_sel_func: F,
}

impl<F> LinkState<F>
where
    F: Fn(&DancingLinks) -> ColumnSelect,
{
    pub fn from_iter(
        iter: impl IntoIterator<Item = impl IntoIterator<Item = usize>>,
        num_columns: usize,
        select_func: F,
    ) -> Self {
        LinkState {
            cur_pos: MoveList::new(),
            links: DancingLinks::from_iter(iter, num_columns),
            col_sel_func: select_func,
        }
    }
    pub fn get_options(&self) -> RowSelect {
        let next_col = (self.col_sel_func)(&self.links);
        match next_col {
            ColumnSelect::Column(col_i) => {
                RowSelect::Rows(self.links.get_rows(col_i))
            }
            ColumnSelect::Fail => RowSelect::Fail,
            ColumnSelect::Complete => RowSelect::Complete,
        }
    }
    pub fn move_to(&mut self, new_pos: MoveList) {
        let mut match_index = None;
        for (i, (cur_val, new_val)) in
            std::iter::zip(self.cur_pos.0.iter(), new_pos.0.iter()).enumerate()
        {
            if cur_val != new_val {
                match_index = Some(i);
                break;
            }
        }
        // if some overlap is not shared
        if let Some(min_index) = match_index {
            // backtracka and uncover necessary columns
            for &uncover_col in self.cur_pos.0[min_index..].iter().rev() {
                self.links.uncover_column(uncover_col);
            }
            // cover necessary columns
            for &cover_col in new_pos.0[min_index..].iter() {
                self.links.cover_column(cover_col);
            }
            // set current position
            self.cur_pos = new_pos;
        } else {
            // check lengths of positions
            match self.cur_pos.0.len().cmp(&new_pos.0.len()) {
                // if lengths are same, positions are same so do nothing
                std::cmp::Ordering::Equal => {}
                // if current position is larger, backtrack
                std::cmp::Ordering::Greater => {
                    for &uncover_col in
                        self.cur_pos.0[new_pos.0.len()..].iter().rev()
                    {
                        self.links.uncover_column(uncover_col);
                    }
                    self.cur_pos = new_pos;
                }
                // if new position is larger, move forward
                std::cmp::Ordering::Less => {
                    for &cover_col in new_pos.0[self.cur_pos.0.len()..].iter() {
                        self.links.cover_column(cover_col);
                    }
                    self.cur_pos = new_pos;
                }
            }
        }
    }
}
