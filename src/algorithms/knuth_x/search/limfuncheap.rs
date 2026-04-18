use super::{FuncHeap, HeapTrait};

/// Struct wrapping generic stable min heap
#[derive(Debug)]
pub struct LimFuncHeap<C, T, F> {
    heap: FuncHeap<C, T, F>,
    limit: Option<usize>,
    fin: bool,
}

// impl<C: Ord, T, F> LimFuncHeap<C, T, F> {
//     pub fn new(cost_func: F, limit: Option<usize>) -> Self {
//         LimFuncHeap {
//             heap: FuncHeap::new(cost_func),
//             limit,
//             fin: false,
//         }
//     }
// }

impl<C: Ord, T, F> LimFuncHeap<C, T, F>
where
    F: Fn(&T) -> C,
{
    pub fn new_single(item: T, cost_func: F, limit: Option<usize>) -> Self {
        let new_heap = FuncHeap::new_single(item, cost_func);
        LimFuncHeap {
            heap: new_heap,
            limit,
            fin: false,
        }
    }
}

impl<C: Ord, T, F> HeapTrait<T> for LimFuncHeap<C, T, F>
where
    F: Fn(&T) -> C,
{
    fn pop(&mut self) -> Option<T> {
        match self.fin {
            true => None,
            false => self.heap.pop(),
        }
    }
    fn push(&mut self, item: T) {
        match (
            if let Some(limit_val) = self.limit {
                self.len() < limit_val
            } else {
                true
            },
            self.fin,
        ) {
            (_, true) => {}
            (true, false) => self.heap.push(item),
            (false, false) => self.fin = true,
        }
    }
    fn len(&self) -> usize {
        self.heap.len()
    }
}
