"""Graph seeding utilities used for initializing a model."""

import collections.abc
import dataclasses
import heapq
import itertools
import math

import numpy
import scipy

import sgrpy

from ..utils import lnbell_est


@dataclasses.dataclass(frozen=True, slots=True)
class GraphSeed:
    graph: sgrpy.graph.SCanonGraph
    weight_guess: float


@dataclasses.dataclass(eq=True, frozen=True, slots=True)
class FragmentData:
    fragment: sgrpy.graph.SCanonGraph
    count: int

    def __lt__(self, other: "FragmentData") -> bool:
        # should return True if worse
        if self.count != other.count:
            return self.count < other.count
        elif len(self.fragment) != len(other.fragment):
            return len(self.fragment) > len(other.fragment)
        return self.fragment > other.fragment


@dataclasses.dataclass(slots=True)
class FragmentHeap:
    _sizelimit: int | None
    _cur_size: int
    _heap_map: dict[int, list[FragmentData]]
    _num_heap: list[int]

    @classmethod
    def new(cls, limit: int | None) -> "FragmentHeap":
        if limit is not None and limit < 0:
            limit = 0
        return FragmentHeap(
            _sizelimit=limit,
            _cur_size=0,
            _heap_map={},
            _num_heap=[],
        )

    def add(self, item: FragmentData) -> bool:
        new_count = item.count

        # add unconditionally if heap is below the limit
        if self._sizelimit is None or self._cur_size < self._sizelimit:
            self._cur_size += 1
            if new_count in self._heap_map:
                self._heap_map[new_count].append(item)
            else:
                heapq.heappush(self._num_heap, new_count)
                self._heap_map[new_count] = [item]
            return True

        # if count is below limit, simply continue
        if new_count < self._num_heap[0]:
            return False

        # add to relevant dict
        self._cur_size += 1
        if new_count in self._heap_map:
            self._heap_map[new_count].append(item)
        else:
            self._heap_map[new_count] = [item]
            heapq.heappush(self._num_heap, new_count)

        # if size is now too big, reduce bracket by one
        cur_min = self._num_heap[0]
        size_lowest_bracket = len(self._heap_map[cur_min])
        if self._cur_size > self._sizelimit + size_lowest_bracket:
            heapq.heappop(self._num_heap)
            del self._heap_map[cur_min]
            self._cur_size -= size_lowest_bracket
            return new_count != cur_min
        return True

    def get_items(self) -> collections.abc.Iterable[FragmentData]:
        return itertools.chain.from_iterable(
            sorted(self._heap_map[i], reverse=True)
            for i in sorted(self._num_heap, reverse=True)
        )

    def get_min(self) -> int | None:
        if len(self._num_heap) == 0:
            return None
        return self._num_heap[0]


def default_naive(
    graphs: collections.abc.Sequence[sgrpy.graph.CGraph],
    num_fragments: int | None,
    canon_method: sgrpy.graph.CanonType,
    max_size: None | int = None,
    min_count: None | int = None,
    use_bell: bool = False,
) -> collections.abc.Iterable[GraphSeed]:
    # print(
    #     """WARNING: `canon_method` should not be used here \
    #        and should be inferred instead"""
    # )

    if min_count is None:
        min_count = 0

    if max_size is not None and max_size < 1:
        raise ValueError(
            f"`max_size` must be positive integer or None (was {max_size})"
        )

    # first, get all atoms
    all_atoms: collections.Counter[int] = collections.Counter()
    atom_est_chain: dict[int, list[float]] = {}
    for graph in graphs:
        glen = len(graph.get_colors())
        color_counter = collections.Counter(graph.get_colors())
        for color, count in color_counter.items():
            # weight_contrib = math.log(count) - lnbell_est(glen)
            if glen == 1 or not use_bell:
                weight_contrib = math.log(count)
            else:
                weight_contrib = (
                    math.log(count) + lnbell_est(glen - 1) - lnbell_est(glen)
                )
            if color in atom_est_chain:
                atom_est_chain[color].append(weight_contrib)
            else:
                atom_est_chain[color] = [weight_contrib]
        all_atoms.update(color_counter.keys())
    atoms_data = [
        FragmentData(
            fragment=sgrpy.graph.CanonGraph.singlet(c, canon_method)[
                0
            ].to_scanongraph(),
            count=count,
        )
        for c, count in all_atoms.items()
    ]

    if num_fragments is not None and len(atoms_data) > num_fragments:
        raise ValueError(f"""Number of atomics ({len(atoms_data)}) greater \
                                 than starting number of tokens \
                                 ({num_fragments})!""")

    heap_limit = (
        None if num_fragments is None else num_fragments - len(atoms_data)
    )
    fragment_heap = FragmentHeap.new(heap_limit)

    weights_est_chain: dict[sgrpy.graph.SCanonGraph, list[float]] = {
        f: atom_est_chain[c]
        for f, c in (
            (c.fragment, c.fragment.to_canongraph().get_colors()[0])
            for c in atoms_data
        )
        if c in atom_est_chain
    }

    cur_tree = sgrpy.graph.GraphTree.from_pins(
        c.fragment.to_canongraph() for c in atoms_data
    )
    cur_size = 2
    while max_size is None or cur_size <= max_size:
        fragment_counts: collections.Counter[sgrpy.graph.SCanonGraph] = (
            collections.Counter()
        )
        for graph in graphs:
            graph_colors = graph.get_colors()
            loc_fragset: dict[sgrpy.graph.SCanonGraph, int] = {}
            graph_bell = lnbell_est(len(graph_colors))
            for locate_map in cur_tree.locate_fragments(graph):
                loc_fragment = cur_tree.get_fragment(locate_map.fragment_id)
                if len(loc_fragment) != cur_size - 1:
                    continue
                cur_nodes = set(locate_map.mapping.as_tuple_int())
                nodes_map = locate_map.mapping.inv().as_tuple()
                node_neighbors: dict[int, tuple[int, list[int]]] = {}
                for n in cur_nodes:
                    neighbors = set(graph.neighbors(n)).difference(cur_nodes)
                    for m in neighbors:
                        if m not in node_neighbors:
                            node_neighbors[m] = (graph_colors[m], [])
                        m_map = nodes_map[n]
                        if m_map is None:
                            raise ValueError("Non-invertible map!")
                        node_neighbors[m][1].append(m_map)

                for new_color, new_neighbor_list in node_neighbors.values():
                    add_operation = sgrpy.graph.AddNode(
                        connections=tuple(sorted(new_neighbor_list)),
                        color=new_color,
                    )
                    new_fragment, _ = loc_fragment.to_canongraph().add_node(
                        add_operation
                    )
                    new_fragment_s = new_fragment.to_scanongraph()
                    if new_fragment_s in loc_fragset:
                        loc_fragset[new_fragment_s] += 1
                    else:
                        loc_fragset[new_fragment_s] = 1
            for fragment, count in loc_fragset.items():
                self_len = len(fragment.to_canongraph().get_colors())
                if self_len == len(graph_colors) or not use_bell:
                    l_contrib = math.log(count)
                else:
                    l_contrib = (
                        lnbell_est(self_len)
                        + lnbell_est(len(graph_colors) - self_len)
                        - graph_bell
                        + math.log(count)
                        # -cur_size * 100 + math.log(count)
                        # cur_size + math.log(count)
                    )
                if fragment in weights_est_chain:
                    weights_est_chain[fragment].append(l_contrib)
                else:
                    weights_est_chain[fragment] = [l_contrib]
            fragment_counts.update(loc_fragset.keys())
        added_new = False
        for new_fragment_d in (
            FragmentData(fragment=s, count=i)
            for s, i in fragment_counts.items()
        ):
            # print(
            #     f"""Proposed new fragment ({new_fragment_d.count}) \
            #         {new_fragment_d.fragment.as_str()}"""
            # )
            if new_fragment_d.count < min_count:
                # print("Rejected due to min count limit!")
                continue
            add_success = fragment_heap.add(new_fragment_d)
            if add_success is True:
                added_new = True
        if not added_new:
            break

        cur_tree = sgrpy.graph.GraphTree.from_pins(
            cg
            for cg in (
                c.fragment.to_canongraph()
                for c in itertools.chain.from_iterable(
                    (atoms_data, fragment_heap.get_items())
                )
            )
            if len(cg.get_colors()) == cur_size
        )
        cur_size += 1

    all_fragments = sorted(
        itertools.chain.from_iterable((atoms_data, fragment_heap.get_items())),
        reverse=True,
    )
    final_tree = sgrpy.graph.GraphTree.from_pins(
        c.fragment.to_canongraph() for c in all_fragments
    )
    # counts = numpy.array(
    #     [c.count for c in all_fragments], dtype=numpy.uintp
    # )
    # sum_counts = numpy.log(numpy.sum(counts))
    # weights = numpy.log(counts) - sum_counts
    weights = numpy.asarray(
        [
            scipy.special.logsumexp(weights_est_chain[d])
            if d in weights_est_chain
            else float("-inf")
            for d in (c.fragment for c in all_fragments)
        ],
        dtype=numpy.float64,
    )
    weights -= scipy.special.logsumexp(weights)

    return (
        GraphSeed(graph=frag, weight_guess=weight)
        for frag, weight in zip(final_tree.iter_pins(), weights, strict=True)
    )
