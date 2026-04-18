use std::collections::BinaryHeap;

use super::queuetrait::ExplicitHeapTrait;

/// Special min heap maintaining insertion order

#[derive(Debug)]
struct HeapItem<C, T> {
    cost: C,
    index: usize,
    item: T,
}

impl<C: PartialEq, T> PartialEq for HeapItem<C, T> {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost && self.index == other.index
    }
}

impl<C: Eq, T> Eq for HeapItem<C, T> {}

impl<C: PartialOrd, T> PartialOrd for HeapItem<C, T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        (&self.cost, self.index)
            .partial_cmp(&(&other.cost, other.index))
            .map(|x| x.reverse())
    }
}

impl<C: Ord, T> Ord for HeapItem<C, T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (&self.cost, self.index)
            .cmp(&(&other.cost, other.index))
            .reverse()
    }
}

#[derive(Debug)]
pub struct MinHeap<C, T> {
    index: usize,
    heap: BinaryHeap<HeapItem<C, T>>,
}

impl<C: Ord, T> MinHeap<C, T> {
    pub fn new() -> Self {
        MinHeap {
            index: 0,
            heap: BinaryHeap::new(),
        }
    }
}

impl<C: Ord, T> ExplicitHeapTrait<C, T> for MinHeap<C, T> {
    fn len(&self) -> usize {
        self.heap.len()
    }
    fn pop(&mut self) -> Option<T> {
        self.heap.pop().map(|x| x.item)
    }
    fn push(&mut self, item: T, cost: C) {
        let heap_item = HeapItem {
            cost,
            index: self.index,
            item,
        };
        self.index += 1;
        self.heap.push(heap_item);
    }
}
