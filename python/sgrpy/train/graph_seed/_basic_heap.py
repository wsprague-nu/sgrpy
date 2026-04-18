"""Heap specification."""

import collections.abc
import dataclasses
import heapq

from sgrpy.iotypes import Comparable


@dataclasses.dataclass(frozen=True, order=True, slots=True)
class _HeapEntry[_CT: Comparable]:
    value: _CT
    order: int


@dataclasses.dataclass(slots=True)
class BasicHeap[_CT: Comparable]:
    _heap: list[_HeapEntry[_CT]]
    _order: int

    @classmethod
    def new(cls) -> "BasicHeap[_CT]":
        heap: list[_HeapEntry[_CT]] = []
        return BasicHeap(_heap=heap, _order=0)

    @classmethod
    def from_iter(cls, iter: collections.abc.Iterable[_CT]) -> "BasicHeap[_CT]":
        heap = [
            _HeapEntry(value=item, order=heap_i)
            for heap_i, item in enumerate(iter)
        ]
        heapq.heapify(heap)
        cur_order = len(heap)
        return BasicHeap(_heap=heap, _order=cur_order)

    def get_min(self) -> _CT:
        return self._heap[0].value

    def replace(self, item: _CT) -> _CT:
        heap_entry = _HeapEntry(value=item, order=self._order)
        self._order += 1
        return heapq.heapreplace(self._heap, heap_entry).value

    def push(self, item: _CT) -> None:
        heap_entry = _HeapEntry(value=item, order=self._order)
        self._order += 1
        heapq.heappush(self._heap, heap_entry)

    def pop(self) -> _CT:
        return heapq.heappop(self._heap).value

    def pushpop(self, item: _CT) -> _CT:
        heap_entry = _HeapEntry(value=item, order=self._order)
        self._order += 1
        return heapq.heappushpop(self._heap, heap_entry).value

    def size(self) -> int:
        return len(self._heap)

    def drain(self) -> collections.abc.Iterable[_CT]:
        for _ in range(self.size()):
            yield self.pop()
        self._order = 0
