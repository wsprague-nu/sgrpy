use super::{ExplicitHeapTrait, HeapTrait, MinHeap};

/// Struct wrapping generic stable min heap
#[derive(Debug)]
pub struct FuncHeap<C, T, F> {
    heap: MinHeap<C, T>,
    cost_func: F,
}

impl<C: Ord, T, F> FuncHeap<C, T, F> {
    pub fn new(cost_func: F) -> Self {
        FuncHeap {
            heap: MinHeap::new(),
            cost_func,
        }
    }
}

impl<C: Ord, T, F> FuncHeap<C, T, F>
where
    F: Fn(&T) -> C,
{
    pub fn new_single(item: T, cost_func: F) -> Self {
        let cost = cost_func(&item);
        let mut fh = FuncHeap::new(cost_func);
        fh.heap.push(item, cost);
        fh
    }
}

impl<C: Ord, T, F> HeapTrait<T> for FuncHeap<C, T, F>
where
    F: Fn(&T) -> C,
{
    fn pop(&mut self) -> Option<T> {
        self.heap.pop()
    }
    fn push(&mut self, item: T) {
        let cost = (self.cost_func)(&item);
        self.heap.push(item, cost);
    }
    fn len(&self) -> usize {
        self.heap.len()
    }
}
