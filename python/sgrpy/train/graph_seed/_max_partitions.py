"""Function for calculating the maximum number of simultaneous covers."""

import collections.abc
import dataclasses

from ._basic_heap import BasicHeap


@dataclasses.dataclass(frozen=True, order=True, slots=True)
class Cover:
    _values: tuple[int, ...]

    @classmethod
    def new(cls, values: collections.abc.Iterable[int]) -> "Cover":
        values_norm = tuple(sorted(frozenset(values)))
        return Cover(_values=values_norm)

    def min_node(self) -> int:
        return self._values[0]

    @property
    def values(self) -> tuple[int, ...]:
        return self._values


@dataclasses.dataclass(frozen=True, slots=True)
class CoverState:
    _cur_cover: int
    _max_future: int
    _covers: tuple[Cover, ...]
    _cover_map: dict[int, set[Cover]]

    def __lt__(self, __value: "CoverState") -> bool:
        return (
            self._cur_cover + self._max_future
            > __value._cur_cover + __value._max_future
        )

    @classmethod
    def new(
        cls, covers: collections.abc.Iterable[Cover], cur_cover: int
    ) -> "CoverState":
        self_covers = tuple(covers)
        cover_map: dict[int, set[Cover]] = {}
        for cover in self_covers:
            for i in cover.values:
                if i not in cover_map:
                    cover_map[i] = {cover}
                else:
                    cover_map[i].add(cover)
        max_future = (
            0
            if len(cover_map) == 0
            else len(cover_map) // min(len(c.values) for c in self_covers)
        )
        return CoverState(
            _cur_cover=cur_cover,
            _max_future=max_future,
            _covers=self_covers,
            _cover_map=cover_map,
        )

    def is_complete(self) -> bool:
        return len(self._covers) == 0

    def cur_count(self) -> int:
        return self._cur_cover

    def get_child(self, remove_cover: Cover) -> "CoverState":
        remove_indices = frozenset(remove_cover.values)
        new_covers = (
            cover
            for cover in self._covers
            if all(v not in remove_indices for v in cover.values)
        )
        return CoverState.new(new_covers, self._cur_cover + 1)

    def get_children(self) -> collections.abc.Iterable["CoverState"]:
        if len(self._covers) == 0:
            return
        select_col = min(self._cover_map, key=lambda x: len(self._cover_map[x]))
        for cover in self._cover_map[select_col]:
            yield self.get_child(cover)


def calc_max_covers(
    covers: collections.abc.Iterable[collections.abc.Iterable[int]],
) -> int:
    cover_iter = (Cover.new(values) for values in covers)
    stateheap = BasicHeap.from_iter((CoverState.new(cover_iter, cur_cover=0),))
    while stateheap.size() > 0:
        cur_state = stateheap.pop()
        children = cur_state.get_children()
        for child in children:
            if child.is_complete():
                return child.cur_count()
            stateheap.push(child)
    return 0
