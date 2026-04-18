use std::collections::BinaryHeap;

/// Simple trait for heap-like objects
pub trait HeapTrait<T> {
    // pop item from the heap-like object
    fn pop(&mut self) -> Option<T>;

    // push item to the heap-like object
    fn push(&mut self, item: T);

    // get size of heap
    fn len(&self) -> usize;
}

impl<T> HeapTrait<T> for Vec<T> {
    fn pop(&mut self) -> Option<T> {
        self.pop()
    }
    fn push(&mut self, item: T) {
        self.push(item)
    }

    fn len(&self) -> usize {
        self.len()
    }
}

impl<T: Ord> HeapTrait<T> for BinaryHeap<T> {
    fn pop(&mut self) -> Option<T> {
        self.pop()
    }
    fn push(&mut self, item: T) {
        self.push(item)
    }
    fn len(&self) -> usize {
        self.len()
    }
}
