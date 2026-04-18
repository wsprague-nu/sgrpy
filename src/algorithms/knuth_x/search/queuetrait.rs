/// Simple trait for queue-like objects
pub trait ExplicitHeapTrait<C, T> {
    // pop item from the queue-like object
    fn pop(&mut self) -> Option<T>;

    // push item to the queue-like object
    fn push(&mut self, item: T, cost: C);

    // get size of queue
    fn len(&self) -> usize;
}

impl<C, T> ExplicitHeapTrait<C, T> for Vec<T> {
    fn pop(&mut self) -> Option<T> {
        self.pop()
    }

    fn push(&mut self, item: T, _cost: C) {
        self.push(item)
    }

    fn len(&self) -> usize {
        self.len()
    }
}
