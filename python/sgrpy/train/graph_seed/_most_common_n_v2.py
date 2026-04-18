"""Functions for seeding cbgraphs based on frequency."""

import dataclasses
import functools
from collections.abc import Iterable, Sequence
from typing import final

from sgrpy.graph import (
    AddNodeT,
    CanonType,
    CBGraph,
    CBGraphC,
    CBGraphCS,
    Extension,
    IndexMapping,
)
from sgrpy.iotypes import Comparable

from ._basic_heap import BasicHeap
from ._max_partitions import calc_max_covers


@final
@dataclasses.dataclass(frozen=True, order=True, slots=True)
class _FragMatch:
    """Object representing a match of a fragment (as child) on a molecule."""

    _mapping_invariant: tuple[int, ...]
    _fragment: CBGraphCS
    _mapping: IndexMapping = dataclasses.field(compare=False)

    @classmethod
    def new(cls, fragment: CBGraphCS, mapping: IndexMapping) -> "_FragMatch":
        invariant = tuple(sorted(mapping.as_tuple_int()))
        return _FragMatch(
            _fragment=fragment, _mapping=mapping, _mapping_invariant=invariant
        )

    @property
    def fragment(self) -> CBGraphCS:
        return self._fragment

    @property
    def invariant(self) -> tuple[int, ...]:
        return self._mapping_invariant

    @property
    def mapping(self) -> IndexMapping:
        return self._mapping

    def get_parents(self) -> Iterable["_FragMatch"]:
        raise NotImplementedError


@final
@dataclasses.dataclass(frozen=True, order=True, slots=True)
class _MatchMap:
    """Object representing a single match of a fragment on a molecule."""

    _mapping: IndexMapping = dataclasses.field(compare=False)
    _mapping_invariant: tuple[int, ...]
    _children: tuple[_FragMatch, ...]

    @classmethod
    def new(
        cls, mapping: IndexMapping, children: Iterable[_FragMatch]
    ) -> "_MatchMap":
        mapping_invariant = tuple(sorted(mapping.as_tuple_int()))
        return _MatchMap(
            _mapping=mapping,
            _children=tuple(sorted(children)),
            _mapping_invariant=mapping_invariant,
        )

    @property
    def children(self) -> tuple[_FragMatch, ...]:
        return self._children

    @property
    def invariant(self) -> tuple[int, ...]:
        return self._mapping_invariant

    @property
    def mapping(self) -> IndexMapping:
        return self._mapping


@final
@dataclasses.dataclass(frozen=True, order=True, slots=True)
class _GraphEntry:
    """Object representing the location(s) of a fragment on a molecule."""

    _graph_id: int
    _matches: tuple[_MatchMap, ...]

    @classmethod
    def new(cls, graph_id: int, matches: Iterable[_MatchMap]) -> "_GraphEntry":
        matches_sorted = tuple(sorted(matches))
        return _GraphEntry(_graph_id=graph_id, _matches=matches_sorted)

    @property
    def graph_id(self) -> int:
        return self._graph_id

    @property
    def matches(self) -> tuple[_MatchMap, ...]:
        return self._matches

    def count_total(self) -> int:
        return len(self._matches)

    def count_mono(self) -> int:
        all_segments = tuple(match.invariant for match in self.matches)
        num_clusters = calc_max_covers(all_segments)
        return num_clusters


@final
@dataclasses.dataclass(frozen=True, order=True, slots=True)
class _FrequencyEntry:
    """Entry representing a graph fragment, its instances, and its children."""

    _children: tuple[_GraphEntry, ...]
    _fragment: CBGraphCS

    @classmethod
    def new(
        cls, fragment: CBGraphCS, children: Iterable[_GraphEntry]
    ) -> "_FrequencyEntry":
        children_sorted = tuple(sorted(children))
        return _FrequencyEntry(_fragment=fragment, _children=children_sorted)

    @property
    def fragment(self) -> CBGraphCS:
        return self._fragment

    @property
    def children(self) -> tuple[_GraphEntry, ...]:
        return self._children

    def count_total(self) -> int:
        return sum(c.count_total() for c in self._children)

    def count_mono(self) -> int:
        return sum(c.count_mono() for c in self._children)


@final
@dataclasses.dataclass(frozen=True, order=True, slots=True)
class _FrequencyEntryHeap[_CT: Comparable]:
    _key: _CT
    _entry: _FrequencyEntry = dataclasses.field(compare=False)

    @classmethod
    def from_entry(
        cls, entry: _FrequencyEntry, key: _CT
    ) -> "_FrequencyEntryHeap[_CT]":
        return _FrequencyEntryHeap(_key=key, _entry=entry)

    @property
    def key(self) -> _CT:
        return self._key

    @property
    def entry(self) -> _FrequencyEntry:
        return self._entry


@final
@dataclasses.dataclass(frozen=True, order=True, slots=True)
class _EntryWeightTup:
    """Weight of individual frequency entry, for search purposes.

    Values are reversed since this is used for a min heap.
    """

    _weight: float | int  # estimated weight of fragment
    _group_count: int  # total groups fragment occurs in
    _item_count: int  # total items fragment occurs in
    _mono_count: int  # estimated non-overlapping occurrences of molecule
    # _raw_count: int  # total overlapping occurrences of molecule

    @classmethod
    def new(
        cls,
        weight: float | int,
        group_count: int,
        item_count: int,
        mono_count: int,
        # raw_count: int,
    ) -> "_EntryWeightTup":
        return _EntryWeightTup(
            _weight=-weight,
            _group_count=-group_count,
            _item_count=-item_count,
            _mono_count=-mono_count,
            # _raw_count=-raw_count,
        )

    @property
    def weight(self) -> int | float:
        return -self._weight

    @property
    def group_count(self) -> int:
        return -self._group_count


def _key_entry_grp(
    graph_weights: Sequence[float | int],
    graph_groups: Sequence[int],
    entry: _FrequencyEntry,
) -> _EntryWeightTup:
    """Return key (lower is better)."""
    graph_ids = frozenset(c.graph_id for c in entry.children)
    group_count = len(frozenset(graph_groups[i] for i in graph_ids))
    item_count = len(graph_ids)
    mono_count = entry.count_mono()
    total_weight = sum(
        c.count_mono() * graph_weights[c.graph_id] for c in entry.children
    )
    return _EntryWeightTup.new(
        weight=total_weight,
        group_count=group_count,
        item_count=item_count,
        mono_count=mono_count,
    )


def _query_fragment_atomic(
    graph: CBGraph, canon_method: CanonType
) -> Iterable[_FragMatch]:
    all_colors = graph.get_colors()
    for i, color in enumerate(all_colors):
        new_graph, _ = CBGraphC.singlet(color, canon_method)
        mapping = IndexMapping.from_seq((i,))
        yield _FragMatch.new(fragment=new_graph.to_sform(), mapping=mapping)


def _locate_new_index(mapping: IndexMapping) -> int:
    for i, m in enumerate(mapping.as_tuple()):
        if m is None:
            return i
    raise ValueError("Failed to locate new index in map")


def _query_fragment(match: _FragMatch, graph: CBGraph) -> Iterable[_FragMatch]:
    select_nodes = match.mapping.as_tuple_int()
    invmap = match.mapping.inv().as_tuple()
    nodes_neighbors: dict[int, list[Extension[int]]] = {}

    # locate new neighbors
    for i in select_nodes:
        index_select = invmap[i]
        if index_select is None:
            raise ValueError("Non-invertible map!")
        for neighbor_ext in graph.neighbors_ext(i):
            if neighbor_ext.idx not in select_nodes:
                if neighbor_ext.idx not in nodes_neighbors:
                    nodes_neighbors[neighbor_ext.idx] = [
                        dataclasses.replace(neighbor_ext, idx=index_select)
                    ]
                else:
                    nodes_neighbors[neighbor_ext.idx].append(
                        dataclasses.replace(neighbor_ext, idx=index_select)
                    )

    node_colors = graph.get_colors()

    # generate new fragments with mappings from neighbor list
    for neighbor_node, connections in nodes_neighbors.items():
        add_op = AddNodeT.new_sort(
            ext=connections, typ=node_colors[neighbor_node]
        )
        new_fragment, new_mapping_s = match.fragment.to_canongraph().add_node(
            add_op
        )
        new_index = _locate_new_index(new_mapping_s)
        new_mapping_shift = new_mapping_s.subst(new_index, len(select_nodes))
        new_mapping = new_mapping_shift.compose(
            match.mapping.append(neighbor_node)
        )

        new_match = _FragMatch.new(
            fragment=new_fragment.to_sform(), mapping=new_mapping
        )

        yield new_match


def _get_atomic_fragments(
    graphs: Iterable[CBGraph], canon_method: CanonType
) -> Iterable[_FrequencyEntry]:
    entry_map: dict[CBGraphCS, dict[int, list[_MatchMap]]] = {}
    for i, graph in enumerate(graphs):
        for fragment_match_atomic in _query_fragment_atomic(
            graph, canon_method
        ):
            atomic_fragment_s = fragment_match_atomic.fragment
            child_matches = list(_query_fragment(fragment_match_atomic, graph))
            if atomic_fragment_s not in entry_map:
                entry_map[atomic_fragment_s] = {}
            if i not in entry_map[atomic_fragment_s]:
                entry_map[atomic_fragment_s][i] = []
            match_map = _MatchMap.new(
                mapping=fragment_match_atomic.mapping, children=child_matches
            )
            entry_map[atomic_fragment_s][i].append(match_map)
    for fragment_s, fragment_entries in entry_map.items():
        graph_entries = tuple(
            _GraphEntry.new(graph_id=graph_id, matches=matchmaps)
            for graph_id, matchmaps in fragment_entries.items()
        )
        freq_entry = _FrequencyEntry.new(fragment_s, graph_entries)
        yield freq_entry


def _get_fragments(
    graphs: Sequence[CBGraph], entry: _FrequencyEntry
) -> Iterable[_FrequencyEntry]:
    entry_map: dict[CBGraphCS, dict[int, list[_MatchMap]]] = {}

    for graph_entry in entry.children:
        cur_graph_id = graph_entry.graph_id
        cur_graph = graphs[cur_graph_id]
        known_matches: set[_FragMatch] = set()
        for graph_match in graph_entry.matches:
            for child_match in graph_match.children:
                if child_match in known_matches:
                    continue
                known_matches.add(child_match)

                cur_fragment = child_match.fragment
                if cur_fragment not in entry_map:
                    entry_map[cur_fragment] = {}

                if cur_graph_id not in entry_map[cur_fragment]:
                    entry_map[cur_fragment][cur_graph_id] = []

                cur_mapping = child_match.mapping

                new_match_map = _MatchMap.new(
                    cur_mapping, _query_fragment(child_match, cur_graph)
                )

                entry_map[cur_fragment][cur_graph_id].append(new_match_map)

    for fragment_s, fragment_entries in entry_map.items():
        graph_entries = tuple(
            _GraphEntry.new(graph_id=graph_id, matches=matchmaps)
            for graph_id, matchmaps in fragment_entries.items()
        )
        freq_entry = _FrequencyEntry.new(fragment_s, graph_entries)
        yield freq_entry


def most_common(
    graphs: Sequence[CBGraph],
    groups: Sequence[int] | None = None,
    weights: Sequence[float | int] | None = None,
    canon_method: CanonType = CanonType.F,
    limit: None | int = None,
    min_count: int | None = None,
    max_size: int | None = None,
) -> Iterable[tuple[CBGraphCS, int | float]]:
    """Obtain most common subgraphs from CBGraph, in order of decreasing weight.

    Atomic fragments are ALWAYS returned, regardless of their frequency, unless
    `limit` or `min_count` are set to zero.

    Parameters
    ----------
    graphs : Sequence[CBGraph]
        Graphs from which to extract subgraphs.
    groups : Sequence[int] | None
        Group indices of graphs (graphs which are part of same group are only
        counted once total for purposes of `min_count`).  Default: all graphs
        are in unique groups.
    weights : Sequence[float | int] | None
        Weights of each graph, contributing to total weight of returned
        subgraph.
    canon_method : CanonType
        Canonization type. Default: `CanonType.F`.
    limit : int | None
        Maximum number of subgraphs to return.  May exceed this value if
        multiple subgraphs have equivalent weight.  Default: all subgraphs
        returned.
    min_count : int | None
        Minimum number of groups each subgraph must occur within.  Default: no
        lower limit.
    max_size : int | None
        Maximum number of vertices a subgraph can possess to be returned.
        Default: no limit.
    """
    # if max size or limit is zero, exit immediately
    if (max_size is not None and max_size <= 0) or (
        limit is not None and limit <= 0
    ):
        return

    # initialize groups and weights to default
    if groups is None:
        groups = tuple(range(len(graphs)))
    if weights is None:
        weights = tuple(1 for _ in graphs)
    if max_size is not None and max_size < 1:
        return

    # ensure number of groups equals number of graphs
    if len(graphs) != len(groups):
        raise ValueError(
            f"Number of groups ({len(groups)} must equal number of graphs "
            f"({len(graphs)})"
        )

    # ensure number of weights equals number of graphs
    if len(graphs) != len(weights):
        raise ValueError(
            f"Number of weights ({len(weights)} must equal number of graphs "
            f"({len(graphs)})"
        )

    # initialize key function for heap
    key_func = functools.partial(_key_entry_grp, weights, groups)

    # sample all atomic fragments from graph set
    atomic_fragments = tuple(_get_atomic_fragments(graphs, canon_method))

    # if no fragments, exit immediately
    if len(atomic_fragments) == 0:
        return

    # create set of accepted fragments
    accepted_fragments = {f.fragment for f in atomic_fragments}
    atomic_fragments_set = frozenset(accepted_fragments)

    # create new priority heap for graph entries
    atom_heap = BasicHeap.from_iter(
        _FrequencyEntryHeap.from_entry(entry, key_func(entry))
        for entry in atomic_fragments
    )
    entry_heap: BasicHeap[_FrequencyEntryHeap[_EntryWeightTup]] = (
        BasicHeap.new()
    )

    # return atom entries
    n_returned = 0
    for atom_entry in atom_heap.drain():
        yield atom_entry.entry.fragment, atom_entry.key.weight
        n_returned += 1
        if min_count is not None and atom_entry.key.group_count < min_count:
            continue
        entry_heap.push(atom_entry)

    # early stopping if limit reached
    if limit is not None and entry_heap.size() >= limit:
        return

    cur_worst = None

    while entry_heap.size() > 0 and (
        limit is None
        or n_returned < limit
        or cur_worst is None
        or cur_worst >= entry_heap.get_min().key
    ):
        # first, pop next fragment entry from stack
        cur_state = entry_heap.pop()
        cur_entry = cur_state.entry
        cur_value = cur_state.key

        # yield fragment entry (but not if atomic, since already returned)
        if cur_entry.fragment not in atomic_fragments_set:
            assert cur_worst is None or cur_worst <= cur_value
            assert min_count is None or cur_value.group_count >= min_count
            yield cur_entry.fragment, cur_value.weight
            n_returned += 1
            cur_worst = cur_value

        # if fragment is at max_size, do not search for children
        if (
            max_size is not None
            and cur_entry.fragment.to_canongraph().nof_nodes() >= max_size
        ):
            continue

        # locate new frequency entries
        for new_entry in _get_fragments(graphs, cur_entry):
            new_fragment = new_entry.fragment
            # if fragment has already been accepted, do not add
            if new_fragment in accepted_fragments:
                continue

            # add fragment to queue
            key_value = key_func(new_entry)

            # if fragment does not satisfy min_count, do not add
            if min_count is not None and key_value.group_count < min_count:
                continue

            accepted_fragments.add(new_fragment)
            new_freq_entry = _FrequencyEntryHeap.from_entry(
                new_entry, key_value
            )
            entry_heap.push(new_freq_entry)
