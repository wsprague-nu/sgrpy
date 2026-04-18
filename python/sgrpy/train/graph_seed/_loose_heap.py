"""Loose heap specification."""

import abc
import collections
import collections.abc
import dataclasses
import heapq
import typing

_T = typing.TypeVar("_T")


class Comparable(typing.Protocol):
    """Protocol for annotating comparable types."""

    @abc.abstractmethod
    def __lt__(self: _T, other: _T) -> bool: ...


_CT = typing.TypeVar("_CT", bound=Comparable)


@dataclasses.dataclass(frozen=True, order=True, slots=True)
class _HeapEntry(typing.Generic[_CT]):
    value: _CT
    entry: int


@dataclasses.dataclass(slots=True)
class _InteriorHeap(typing.Generic[_CT]):
    _heap: list[_HeapEntry[_CT]]
    _limit: int
    _order: int

    @classmethod
    def with_limit(cls, limit: None | int) -> "_InteriorHeap[_CT]":
        heap: list[_HeapEntry[_CT]] = []
        self_limit = -1 if limit is None else limit
        return _InteriorHeap(_heap=heap, _limit=self_limit, _order=0)

    def get_min(self) -> None | _CT:
        return None if len(self._heap) == 0 else self._heap[0].value

    def _heappush(self, item: _CT) -> None:
        new_item = _HeapEntry(item, self._order)
        self._order -= 1
        heapq.heappush(self._heap, new_item)

    def _reject_items(self) -> collections.deque[_CT]:
        # assumptions of this function: heap size is not zero
        cur_min = self._heap[0]
        rejects: collections.deque[_CT] = collections.deque()
        while len(self._heap) > 0 and not (cur_min.value < self._heap[0].value):
            rejects.append(heapq.heappop(self._heap).value)

        return rejects

    def push(self, item: _CT) -> None | collections.abc.Iterable[_CT]:
        """Push new item, return overflow items if exceeding limit."""
        # add item if heap is not full
        if self._limit < 0 or len(self._heap) < self._limit:
            self._heappush(item)
            return None
        heap_min = None if len(self._heap) == 0 else self._heap[0].value
        # if item does not fit into heap, reject it
        if heap_min is None or item < heap_min:
            return (item,)

        # reject min heap items
        rejects = self._reject_items()

        # if item supersedes minimum heap item, push
        if heap_min < item:
            new_item = _HeapEntry(item, self._order)
            self._order -= 1
            heapq.heappush(self._heap, new_item)
        # else add to rejects
        else:
            rejects.appendleft(item)

        return reversed(rejects)

    def size(self) -> int:
        return len(self._heap)

    def drain(self) -> collections.abc.Iterable[_CT]:
        drain_list: list[_CT] = []
        temp_queue: list[_CT] = []
        for _ in range(len(self._heap)):
            cur_val = heapq.heappop(self._heap).value
            temp_queue.append(cur_val)
        drain_list.extend(reversed(temp_queue))
        return drain_list


@dataclasses.dataclass(slots=True)
class LooseHeap(typing.Generic[_CT]):
    _heap: _InteriorHeap[_CT]  # heap of size equal to or below limit
    _limit: int  # heap size limit
    _wait_queue: list[_CT]  # equivalent items taking heap size above limit

    @classmethod
    def with_limit(cls, limit: None | int) -> "LooseHeap[_CT]":
        heap: _InteriorHeap[_CT] = _InteriorHeap.with_limit(limit)
        wait_queue: list[_CT] = []
        self_limit = -1 if limit is None else limit
        return LooseHeap(_heap=heap, _limit=self_limit, _wait_queue=wait_queue)

    @classmethod
    def from_iter(
        cls, iter: collections.abc.Iterable[_CT], limit: None | int
    ) -> "LooseHeap[_CT]":
        heap: LooseHeap[_CT] = LooseHeap.with_limit(limit)
        for item in iter:
            heap.push(item)
        return heap

    def get_min(self) -> None | _CT:
        if len(self._wait_queue) > 0:
            return self._wait_queue[-1]
        return self._heap.get_min()

    def size(self) -> int:
        return self._heap.size() + len(self._wait_queue)

    def push(self, item: _CT) -> None:
        # check if item supersedes wait queue
        queue_upper_bound = (
            self._wait_queue[0] if len(self._wait_queue) > 0 else None
        )
        # if item supersedes queue upper bound, push to heap
        if queue_upper_bound is None or queue_upper_bound < item:
            rejects = self._heap.push(item)
            # if there are rejects, clear queue
            if rejects is not None:
                self._wait_queue.clear()
                # if the size of the internal heap is under limit, make queue
                if self._heap.size() < self._limit:
                    self._wait_queue.extend(rejects)
        # if queue upper bound does not supersede item, add item to queue
        elif not (item < queue_upper_bound):
            self._wait_queue.append(item)

    def drain(self) -> collections.abc.Iterable[_CT]:
        # draining items from heap
        drain_list = list(self._heap.drain())
        drain_list.extend(self._wait_queue)
        self._wait_queue.clear()
        return drain_list
