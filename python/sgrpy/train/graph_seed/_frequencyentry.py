"""Data structure for subgraph detection and tracking."""

import collections.abc
import dataclasses

import sgrpy

from ._max_partitions import calc_max_covers


def calc_clusters(
    covers: collections.abc.Iterable[collections.abc.Iterable[int]],
) -> int:
    cluster_sets: list[set[int]] = []

    merged = False
    # create initial clusters
    for new_cover in covers:
        cover_set = set(new_cover)
        for compare_set in cluster_sets:
            if len(cover_set.intersection(compare_set)) > 0:
                compare_set.update(cover_set)
                merged = True
                break
        if not merged:
            cluster_sets.append(cover_set)
        merged = False

    # iterate until number of clusters is stable
    prev_clusters: list[set[int]] | None = None
    cur_clusters = cluster_sets
    while prev_clusters is None or len(cur_clusters) < len(prev_clusters):
        prev_clusters = cur_clusters
        cur_clusters = []
        merged = False
        for cluster in prev_clusters:
            for compare_set in cur_clusters:
                if len(cluster.intersection(compare_set)) > 0:
                    compare_set.update(cluster)
                    merged = True
                    break
            if not merged:
                cur_clusters.append(cluster)
            merged = False

    return len(cur_clusters)


@dataclasses.dataclass(frozen=True, order=True, slots=True)
class FragmentMatch:
    _mapping_invariant: tuple[int, ...]
    _fragment: sgrpy.graph.SCanonGraph
    _mapping: sgrpy.graph.IndexMapping = dataclasses.field(compare=False)

    @classmethod
    def new(
        cls,
        fragment: sgrpy.graph.SCanonGraph,
        mapping: sgrpy.graph.IndexMapping,
    ) -> "FragmentMatch":
        invariant = tuple(sorted(mapping.as_tuple_int()))
        return FragmentMatch(
            _fragment=fragment, _mapping=mapping, _mapping_invariant=invariant
        )

    @property
    def fragment(self) -> sgrpy.graph.SCanonGraph:
        return self._fragment

    @property
    def invariant(self) -> tuple[int, ...]:
        return self._mapping_invariant

    @property
    def mapping(self) -> sgrpy.graph.IndexMapping:
        return self._mapping

    def get_parents(self) -> collections.abc.Iterable["FragmentMatch"]:
        canongraph = self.fragment.to_canongraph()
        if len(canongraph.get_colors()) == 1:
            return
        cur_mapping = self.mapping
        for sub_i in range(len(canongraph.get_colors())):
            sub_op = sgrpy.graph.SubNode(sub_i)
            parent_fragment, parent_mapping = canongraph.sub_node(sub_op)
            if parent_fragment.cgraph.nof_components() > 1:
                continue
            full_mapping = parent_mapping.compose(cur_mapping)
            pass
            yield FragmentMatch.new(
                parent_fragment.to_scanongraph(), mapping=full_mapping
            )


@dataclasses.dataclass(frozen=True, order=True, slots=True)
class MatchMap:
    _mapping: sgrpy.graph.IndexMapping = dataclasses.field(compare=False)
    _mapping_invariant: tuple[int, ...]
    _children: tuple[FragmentMatch, ...]

    @classmethod
    def new(
        cls,
        mapping: sgrpy.graph.IndexMapping,
        children: collections.abc.Iterable[FragmentMatch],
    ) -> "MatchMap":
        mapping_invariant = tuple(sorted(mapping.as_tuple_int()))
        return MatchMap(
            _mapping=mapping,
            _children=tuple(sorted(children)),
            _mapping_invariant=mapping_invariant,
        )

    @property
    def children(self) -> tuple[FragmentMatch, ...]:
        return self._children

    @property
    def invariant(self) -> tuple[int, ...]:
        return self._mapping_invariant

    @property
    def mapping(self) -> sgrpy.graph.IndexMapping:
        return self._mapping


@dataclasses.dataclass(frozen=True, order=True, slots=True)
class GraphEntry:
    _graph_id: int
    _matches: tuple[MatchMap, ...]

    @classmethod
    def new(
        cls, graph_id: int, matches: collections.abc.Iterable[MatchMap]
    ) -> "GraphEntry":
        matches_sorted = tuple(sorted(matches))
        return GraphEntry(_graph_id=graph_id, _matches=matches_sorted)

    @property
    def graph_id(self) -> int:
        return self._graph_id

    @property
    def matches(self) -> tuple[MatchMap, ...]:
        return self._matches

    def count_total(self) -> int:
        return len(self._matches)

    def count_mono(self) -> int:
        all_segments = tuple(match.invariant for match in self.matches)
        num_clusters = calc_max_covers(all_segments)
        return num_clusters


@dataclasses.dataclass(frozen=True, order=True, slots=True)
class FrequencyEntry:
    _children: tuple[GraphEntry, ...]
    _fragment: sgrpy.graph.SCanonGraph

    @classmethod
    def new(
        cls,
        fragment: sgrpy.graph.SCanonGraph,
        children: collections.abc.Iterable[GraphEntry],
    ) -> "FrequencyEntry":
        return FrequencyEntry(
            _fragment=fragment, _children=tuple(sorted(children))
        )

    @property
    def fragment(self) -> sgrpy.graph.SCanonGraph:
        return self._fragment

    @property
    def children(self) -> tuple[GraphEntry, ...]:
        return self._children

    def count_total(self) -> int:
        return sum(g.count_total() for g in self._children)

    def count_mono(self) -> int:
        return sum(g.count_mono() for g in self._children)

    def graph_ids(self) -> frozenset[int]:
        return frozenset(g.graph_id for g in self._children)

    def graphs_mono(self) -> dict[int, int]:
        return {g.graph_id: g.count_mono() for g in self._children}

    def is_dependent(self) -> bool:
        dep_children: None | frozenset[sgrpy.graph.SCanonGraph] = None
        for graphmatch in self.children:
            for match in graphmatch.matches:
                if dep_children is None:
                    dep_children = frozenset(
                        child.fragment for child in match.children
                    )
                else:
                    dep_children = dep_children.intersection(
                        child.fragment for child in match.children
                    )
                if len(dep_children) == 0:
                    return False
        return dep_children is not None
