"""Method for seeding graphs based on frequency."""

import abc
import dataclasses
import functools
import typing
from collections.abc import Container, Iterable, Sequence

import sgrpy

from ._basic_heap import BasicHeap
from ._frequencyentry import FragmentMatch, FrequencyEntry, GraphEntry, MatchMap

_T = typing.TypeVar("_T")


class Comparable(typing.Protocol):
    """Protocol for annotating comparable types."""

    @abc.abstractmethod
    def __lt__(self: _T, __value: _T) -> bool: ...


_CT = typing.TypeVar("_CT", bound=Comparable)


def query_fragment_atomic(
    graph: sgrpy.graph.CGraph,
    canon_method: sgrpy.graph.CanonType = sgrpy.graph.CanonType.F,
) -> Iterable[FragmentMatch]:
    all_colors = graph.get_colors()
    for i, color in enumerate(all_colors):
        new_graph, _ = sgrpy.graph.CanonGraph.from_bonds(
            (), (color,), canon_method
        )
        mapping = sgrpy.graph.IndexMapping.from_seq((i,))
        yield FragmentMatch.new(
            fragment=new_graph.to_scanongraph(), mapping=mapping
        )


def _locate_new_index(mapping: sgrpy.graph.IndexMapping) -> int:
    for i, m in enumerate(mapping.as_tuple()):
        if m is None:
            return i
    raise ValueError("Failed to locate new index in map")


def query_fragment(
    match: FragmentMatch,
    graph: sgrpy.graph.CGraph,
) -> Iterable[FragmentMatch]:
    select_nodes = match.mapping.as_tuple_int()
    invmap = match.mapping.inv().as_tuple()
    nodes_neighbors: dict[int, list[int]] = {}

    # locate new neighbors
    for i in select_nodes:
        index_select = invmap[i]
        if index_select is None:
            raise ValueError("Non-invertible map!")
        for neighbor in graph.neighbors(i):
            if neighbor not in select_nodes:
                if neighbor not in nodes_neighbors:
                    nodes_neighbors[neighbor] = [index_select]
                else:
                    nodes_neighbors[neighbor].append(index_select)

    node_colors = graph.get_colors()

    # generate new fragments with mappings from neighbor list
    for neighbor_node, connections in nodes_neighbors.items():
        add_operation = sgrpy.graph.AddNode(
            connections=tuple(sorted(connections)),
            color=node_colors[neighbor_node],
        )
        new_fragment, new_mapping_s = match.fragment.to_canongraph().add_node(
            add_operation
        )
        new_index = _locate_new_index(new_mapping_s)
        new_mapping_shift = new_mapping_s.subst(new_index, len(select_nodes))
        new_mapping = new_mapping_shift.compose(
            match.mapping.append(neighbor_node)
        )

        new_match = FragmentMatch.new(
            fragment=new_fragment.to_scanongraph(), mapping=new_mapping
        )

        yield new_match


def get_atomic_fragments(
    graphs: Iterable[sgrpy.graph.CGraph],
    canon_method: sgrpy.graph.CanonType = sgrpy.graph.CanonType.F,
) -> Iterable[FrequencyEntry]:
    entry_map: dict[
        sgrpy.graph.SCanonGraph,
        dict[int, list[MatchMap]],
    ] = {}
    for i, graph in enumerate(graphs):
        for fragment_match_atomic in query_fragment_atomic(graph, canon_method):
            atomic_fragment_s = fragment_match_atomic.fragment
            child_matches: list[FragmentMatch] = []
            for child_match in query_fragment(fragment_match_atomic, graph):
                child_matches.append(child_match)
            if atomic_fragment_s not in entry_map:
                entry_map[atomic_fragment_s] = {}
            if i not in entry_map[atomic_fragment_s]:
                entry_map[atomic_fragment_s][i] = []
            match_map = MatchMap.new(
                mapping=fragment_match_atomic.mapping,
                children=child_matches,
            )
            entry_map[atomic_fragment_s][i].append(match_map)

    for fragment_s, fragment_entries in entry_map.items():
        graph_entries = tuple(
            GraphEntry.new(graph_id=graph_id, matches=matchmaps)
            for graph_id, matchmaps in fragment_entries.items()
        )
        freq_entry = FrequencyEntry.new(
            fragment_s,
            graph_entries,
        )
        yield freq_entry


def get_fragments(
    graphs: Sequence[sgrpy.graph.CGraph],
    entry: FrequencyEntry,
    known_fragments: Container[sgrpy.graph.SCanonGraph],
) -> Iterable[FrequencyEntry]:
    entry_map: dict[
        sgrpy.graph.SCanonGraph,
        dict[int, list[MatchMap]],
    ] = {}

    for graph_entry in entry.children:
        cur_graph_id = graph_entry.graph_id
        cur_graph = graphs[cur_graph_id]
        known_matches: set[FragmentMatch] = set()
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

                new_match_map = MatchMap.new(
                    cur_mapping, query_fragment(child_match, cur_graph)
                )

                entry_map[cur_fragment][cur_graph_id].append(new_match_map)

    for fragment_s, fragment_entries in entry_map.items():
        graph_entries = tuple(
            GraphEntry.new(graph_id=graph_id, matches=matchmaps)
            for graph_id, matchmaps in fragment_entries.items()
        )
        freq_entry = FrequencyEntry.new(
            fragment_s,
            graph_entries,
        )
        yield freq_entry


@dataclasses.dataclass(frozen=True, order=True, slots=True)
class HeapKey(typing.Generic[_CT]):
    value: _CT

    @classmethod
    def new(cls, key: _CT) -> "HeapKey[_CT]":
        return HeapKey(value=key)


@dataclasses.dataclass(frozen=True, order=True, slots=True)
class FrequencyEntryHeap(typing.Generic[_CT]):
    _key: HeapKey[_CT]
    _entry: FrequencyEntry = dataclasses.field(compare=False)

    @classmethod
    def from_entry(
        cls, entry: FrequencyEntry, key: _CT
    ) -> "FrequencyEntryHeap[_CT]":
        hkey = HeapKey.new(key)
        return FrequencyEntryHeap(_key=hkey, _entry=entry)

    @property
    def key(self) -> HeapKey[_CT]:
        return self._key

    @property
    def entry(self) -> FrequencyEntry:
        return self._entry


def key_entry(entry: FrequencyEntry) -> int:
    return -entry.count_mono()


def key_entry_grp(
    graph_weights: Sequence[float | int],
    graph_groups: Sequence[int],
    entry: FrequencyEntry,
) -> tuple[float | int, int]:
    """Return key (lower is better)."""
    mono_counts = entry.graphs_mono()
    mults = sum(
        (c * w)
        for c, w in sorted(
            (count, graph_weights[graph_id])
            for graph_id, count in mono_counts.items()
        )
    )
    grp_counts = len(frozenset(graph_groups[gi] for gi in entry.graph_ids()))
    return -mults, -grp_counts


def most_common(
    graphs: Sequence[sgrpy.graph.CGraph],
    canon_method: sgrpy.graph.CanonType = sgrpy.graph.CanonType.F,
    limit: None | int = None,
    exc_dependent: bool = False,
    min_count: int | None = None,
) -> Iterable[tuple[sgrpy.graph.SCanonGraph, int]]:
    """Yield most common subgraphs, according to maximum independent count.

    Atomic subgraphs are always yielded first, even if have a lower total count
    than some larger subgraphs.

    Parameters
    ----------
    graphs : Sequence[sgrpy.graph.CGraph]
        Graphs from which to mine subgraphs.
    canon_method : sgrpy.graph.CanonType
        Type of graph canonicalization to use for returned subgraphs
        (default: F).
    limit : None | int
        Limit on number of returned values (default: None). May be exceeded if
        there are several subgraphs with the same count.
    exc_dependent : bool
        Whether to exclude subgraphs which only occur as a part of a larger
        subgraph.
    min_count : None | int
        Minimum number of occurrences required for returned fragments (default:
        None).

    Yields
    ------
    tuple[sgrpy.graph.SCanonGraph,int]
        A tuple where the first element is the subgraph, and the
        second element is the count of that subgraph.

    Notes
    -----
    The use of "maximum independent count" refers to the practice of, when
    dealing with graphs where a subgraph can be mapped to overlapping subsets of
    nodes, to calculate the "total count" for that graph by considering the
    maximum number of simultaneous non-overlapping mappings for that subgraph.
    """
    # sample all atomic fragments from graph set
    atomic_fragments = tuple(get_atomic_fragments(graphs, canon_method))

    # if no fragments, exit
    if len(atomic_fragments) == 0 or limit == 0:
        return

    # create set of accepted fragments
    accepted_fragments = {f.fragment for f in atomic_fragments}
    atomic_fragments_set = frozenset(accepted_fragments)

    # create new priority heap for graph entries
    atom_heap = BasicHeap.from_iter(
        FrequencyEntryHeap.from_entry(entry, key_entry(entry))
        for entry in atomic_fragments
    )
    entry_heap: BasicHeap[FrequencyEntryHeap[int]] = BasicHeap.new()

    # return atom entries
    n_returned = 0
    for atom_entry in atom_heap.drain():
        yield atom_entry.entry.fragment, -atom_entry.key.value
        entry_heap.push(atom_entry)
        n_returned += 1

    # return if limit is reached
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
        cur_count = -cur_state.key.value

        # if min_count has been reached, end the iteration immediately
        if min_count is not None and cur_count < min_count:
            return

        # yield fragment entry (but not if atomic, since already returned)
        if cur_entry.fragment not in atomic_fragments_set and not (
            exc_dependent and cur_entry.is_dependent()
        ):
            yield cur_entry.fragment, cur_count
            n_returned += 1
            cur_worst = cur_state.key

        # locate new frequency entries
        for new_entry in get_fragments(
            graphs, cur_entry, frozenset(accepted_fragments)
        ):
            new_fragment = new_entry.fragment
            # if fragment has already been accepted, do not add
            if new_fragment in accepted_fragments:
                continue

            # add fragment to queue
            key_value = key_entry(new_entry)
            accepted_fragments.add(new_fragment)
            entry_heap.push(FrequencyEntryHeap.from_entry(new_entry, key_value))


def most_common_grouped(
    graphs: Sequence[sgrpy.graph.CGraph],
    groups: Sequence[int] | None = None,
    weights: Sequence[float | int] | None = None,
    canon_method: sgrpy.graph.CanonType = sgrpy.graph.CanonType.F,
    limit: None | int = None,
    exc_dependent: bool = False,
    min_count: int | None = None,
    max_size: int | None = None,
) -> Iterable[tuple[sgrpy.graph.SCanonGraph, int | float]]:
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

    key_func = functools.partial(key_entry_grp, weights, groups)

    # sample all atomic fragments from graph set
    atomic_fragments = tuple(get_atomic_fragments(graphs, canon_method))

    # if no fragments, exit
    if len(atomic_fragments) == 0 or limit == 0:
        return

    # create set of accepted fragments
    accepted_fragments = {f.fragment for f in atomic_fragments}
    atomic_fragments_set = frozenset(accepted_fragments)

    # create new priority heap for graph entries
    atom_heap = BasicHeap.from_iter(
        FrequencyEntryHeap.from_entry(entry, key_func(entry))
        for entry in atomic_fragments
    )
    entry_heap: BasicHeap[FrequencyEntryHeap[tuple[int | float, int]]] = (
        BasicHeap.new()
    )

    # return atom entries
    n_returned = 0
    for atom_entry in atom_heap.drain():
        yield atom_entry.entry.fragment, -atom_entry.key.value[0]
        entry_heap.push(atom_entry)
        n_returned += 1

    # return if limit is reached
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
        cur_count = -cur_state.key.value[0]

        # yield fragment entry (but not if atomic, since already returned)
        if cur_entry.fragment not in atomic_fragments_set and not (
            exc_dependent and cur_entry.is_dependent()
        ):
            yield cur_entry.fragment, cur_count
            n_returned += 1
            cur_worst = cur_state.key

        # if fragment is same as max_size, do not search for children
        if (
            max_size is not None
            and cur_entry.fragment.to_canongraph().nof_nodes() >= max_size
        ):
            continue

        # locate new frequency entries
        for new_entry in get_fragments(
            graphs, cur_entry, frozenset(accepted_fragments)
        ):
            new_fragment = new_entry.fragment
            # if fragment has already been accepted, do not add
            if new_fragment in accepted_fragments:
                continue

            # add fragment to queue
            key_value = key_func(new_entry)

            # if fragment does not satisfy min_count, do not add
            if min_count is not None and -key_value[1] < min_count:
                continue

            accepted_fragments.add(new_fragment)
            new_freq_entry = FrequencyEntryHeap.from_entry(new_entry, key_value)
            entry_heap.push(new_freq_entry)
