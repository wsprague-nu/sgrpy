"""Fragment model datastructure."""

import collections
import collections.abc
import dataclasses
import heapq
import itertools
import logging

import numpy
import numpy.typing
import scipy
import tqdm

import sgrpy
import sgrpy.dl_rs
from sgrpy.iotypes import JSON, json_to_dict, json_to_list

from ._observation import Observation
from .graph_seed import bell_naive
from .utils import (
    est_num_graphs,
    lndfac_unchecked,
    lnfac_unchecked,
    normalize_int_log,
    normalize_log_log,
)


def calc_ac_factor(
    ccounts: collections.abc.Sequence[int], v_remaining: int
) -> float:
    ccounts_total = sum(ccounts)
    composition_cost = sum(map(lnfac_unchecked, ccounts))
    combination_cost = lnfac_unchecked(ccounts_total + v_remaining)
    return composition_cost - combination_cost


def calc_as_factor(
    ecounts: collections.abc.Sequence[int], etotal: int
) -> float:
    return lndfac_unchecked(etotal - sum(ecounts))


@dataclasses.dataclass(frozen=True, slots=True)
class FragmentWeighted:
    fragment: sgrpy.graph.SCanonGraph
    weight: numpy.float64

    def __lt__(self, other: "FragmentWeighted") -> bool:
        # should return True if worse
        if self.weight != other.weight:
            return bool(self.weight < other.weight)
        elif len(self.fragment) != len(other.fragment):
            return len(self.fragment) < len(other.fragment)
        return self.fragment > other.fragment


@dataclasses.dataclass(frozen=True, slots=True)
class FragmentPruneCandidate:
    fragment: sgrpy.graph.CanonGraph
    weight: numpy.float64
    loss: numpy.float64

    def is_atomic(self) -> bool:
        return len(self.fragment.get_colors()) == 1

    def __lt__(self, other: "FragmentPruneCandidate") -> bool:
        # should return True if better
        self_atomic = self.is_atomic()
        other_atomic = other.is_atomic()
        if self_atomic != other_atomic:
            return self_atomic > other_atomic
        if self.loss != other.loss:
            return bool(self.loss < other.loss)
        if self.weight != other.weight:
            return bool(self.weight > other.weight)
        self_len = len(self.fragment.get_colors())
        other_len = len(other.fragment.get_colors())
        if self_len != other_len:
            return self_len > other_len
        return self.fragment.to_scanongraph() < other.fragment.to_scanongraph()


@dataclasses.dataclass(frozen=True, slots=True)
class FragModel:
    """Fragment weight and detection model, used for partitioning CGraphs."""

    _fragtree: sgrpy.graph.GraphTree
    _weights: numpy.typing.NDArray[numpy.float64]

    @classmethod
    def from_tree(
        cls,
        fragment_tree: sgrpy.graph.GraphTree,
        weights: numpy.typing.NDArray[numpy.float64],
    ) -> "FragModel":
        # sanity check
        if fragment_tree.nof_pinned() != len(weights):
            raise ValueError(
                f"Size of fragment tree ({fragment_tree.nof_pinned()}) must "
                f"match length of weight vector ({len(weights)})"
            )
        final_weights = normalize_log_log(weights)
        return FragModel(_fragtree=fragment_tree, _weights=final_weights)

    @classmethod
    def new(
        cls,
        fragments: collections.abc.Iterable[sgrpy.graph.CanonGraph],
        weights: numpy.typing.NDArray[numpy.float64],
    ) -> "FragModel":
        fragtree = sgrpy.graph.GraphTree.from_pins(fragments)
        return FragModel.from_tree(fragtree, weights)

    @property
    def weights(self) -> numpy.typing.NDArray[numpy.float64]:
        return self._weights.copy()

    @property
    def fragtree(self) -> sgrpy.graph.GraphTree:
        return self._fragtree

    def nof_fragments(self) -> int:
        return self._weights.size

    def nof_fragments_atomic(self) -> int:
        return sum(
            1
            for x in (
                self._fragtree.get_fragment(i)
                for i in range(self._fragtree.nof_pinned())
            )
            if len(x.to_canongraph().get_colors()) == 1
        )

    def calc_color_weights(
        self,
    ) -> collections.abc.Mapping[int, numpy.float64]:
        """Get mapping of colors to their weights."""
        # tally counts of each color
        color_counter = collections.Counter(
            itertools.chain.from_iterable(
                fragment.to_canongraph().get_colors()
                for fragment in self._fragtree.iter_pins()
            )
        )

        # move colors into tuple form for order consistency
        color_tuples = tuple(
            (color, count) for color, count in color_counter.items()
        )

        # convert color counts to an integer vectors
        color_countvec = numpy.asarray(
            tuple(count for _, count in color_tuples), dtype=numpy.uintp
        )

        # normalize color integer vector
        color_weightvec = normalize_int_log(color_countvec)

        # assemble dict mapping color to weight
        color_weights: dict[int, numpy.float64] = {
            color: prob
            for (color, _), prob in zip(
                color_tuples, color_weightvec, strict=True
            )
        }

        return color_weights

    def calc_ll_fragment(
        self,
        i: int,
        color_weights: None
        | collections.abc.Mapping[int, numpy.float64] = None,
        use_struc_fac: bool = False,
    ) -> numpy.float64:
        """Calculate loglikelihood of fragment with index i."""
        # if no color weights provided, use self-calculated ones
        if color_weights is None:
            color_weights = self.calc_color_weights()

        # get target fragment
        fragment = self._fragtree.get_fragment(i).to_canongraph()

        color_seq = fragment.get_colors()

        # loglikelihood from the color permutation
        color_ll = numpy.sum(tuple(color_weights[c] for c in color_seq))

        # correction factor for graph structure (asymptotic)
        if use_struc_fac:
            graph_ll = numpy.float64(-est_num_graphs(len(color_seq)))
            color_ll += graph_ll

        return color_ll

    def calc_ll(
        self,
        color_weights: None
        | collections.abc.Mapping[int, numpy.float64] = None,
        use_struc_fac: bool = False,
    ) -> numpy.float64:
        """Calculate self-loglikelihood of model."""
        # if no color weights provided, use self-calculated ones
        if color_weights is None:
            color_weights = self.calc_color_weights()

        return numpy.sum(
            tuple(
                self.calc_ll_fragment(i, color_weights, use_struc_fac)
                for i in range(self.nof_fragments())
            )
        )

    def partition(
        self,
        graph: sgrpy.graph.CGraph,
        limit: None | int,
        max_iter: None | int,
        max_heap: None | int,
        useac: bool = False,
    ) -> collections.abc.Iterable[Observation]:
        """Partition graph based on internal model.

        Partitioning currently calls Rust code from `dl_rs`.

        Parameters
        ----------
        graph: CGraph
            Graph to be partitioned.
        limit: None | int
            Maximum number of partitions to return.
        max_iter : None | int
            Maximum number of iterations to search.
        max_heap : None | int
            Maximum size of the search tree.
        useac: bool
            Include generic graph correction factors in returned likelihood
            (default: False).

        Yields
        ------
        Observation
            Observation datastructure, containing both partition and likelihood
            information.
        """
        # locate fragment matches
        all_matches = tuple(self._fragtree.locate_fragments(graph))
        match_covers = tuple(
            match.mapping.as_tuple_int() for match in all_matches
        )

        # check for partial cover
        all_colors = frozenset(itertools.chain.from_iterable(match_covers))
        if all_colors != frozenset(range(graph.nof_nodes())):
            return

        match_weights = tuple(
            -self._weights[match.fragment_id] for match in all_matches
        )

        # get fragment labels
        labels = tuple(match.fragment_id for match in all_matches)

        # get vertex counts
        vertex_total = graph.nof_nodes()
        vertex_counts = tuple(
            self._fragtree.get_fragment(fragment_id).to_canongraph().nof_nodes()
            for fragment_id in labels
        )

        # get edge counts
        edge_total = graph.nof_edges()
        edge_counts = tuple(
            self._fragtree.get_fragment(fragment_id).to_canongraph().nof_edges()
            for fragment_id in labels
        )

        cover_enum: list[list[int]]
        if not useac:
            cover_enum = (
                sgrpy.dl_rs.get_top(match_covers, match_weights)
                if limit is None
                else sgrpy.dl_rs.get_topn(
                    match_covers, match_weights, limit, max_iter, max_heap
                )
            )
        else:
            if limit is None:
                logging.warn(
                    "no limit and `useac` not yet supported; "
                    "setting limit to 64"
                )
                limit = 64
            cover_enum = sgrpy.dl_rs.get_topn_sc(
                match_covers,
                match_weights,
                edge_counts,
                edge_total,
                labels,
                vertex_total,
                limit,
                max_iter,
                max_heap,
            )

        for covers in cover_enum:
            known_matches = frozenset(all_matches[x] for x in covers)
            frag_counts = numpy.zeros(self.nof_fragments(), dtype=numpy.uintp)
            for match_id in covers:
                frag_counts[all_matches[match_id].fragment_id] += 1
            temp_weights = self._weights.copy()
            temp_weights[
                (
                    (self._weights == numpy.float64("-inf"))
                    & (frag_counts == numpy.uintp(0))
                )
            ] = numpy.float64(0)
            loglikelihood = numpy.dot(temp_weights, frag_counts)

            if useac:
                local_labels = tuple(
                    collections.Counter(labels[p] for p in covers).values()
                )
                vertices = tuple(vertex_counts[p] for p in covers)
                ac_factor = calc_ac_factor(
                    local_labels, vertex_total - sum(vertices)
                )
                edges = tuple(edge_counts[p] for p in covers)
                as_factor = calc_as_factor(edges, edge_total)
                loglikelihood -= ac_factor + as_factor
            yield Observation(matches=known_matches, weight=loglikelihood)

    def with_weights(
        self, weights: numpy.typing.NDArray[numpy.float64]
    ) -> "FragModel":
        """Return model with same structure but new weights (normalized)."""
        return FragModel.from_tree(fragment_tree=self.fragtree, weights=weights)

    def with_subset(
        self, select_frag: collections.abc.Iterable[int]
    ) -> "FragModel":
        """Return new model, retaining only selected fragments."""
        to_retain = tuple(sorted(frozenset(select_frag)))
        frag_gen = (
            self.fragtree.get_fragment(i).to_canongraph() for i in to_retain
        )
        new_weights = numpy.take(self._weights, to_retain)
        return FragModel.new(fragments=frag_gen, weights=new_weights)

    def prune(
        self,
        graphs: collections.abc.Sequence[sgrpy.graph.CGraph],
        final_num: int,
        use_ac: bool = False,
    ) -> "FragModel":
        # if final_num > self._weights.size or final_num < 0:
        #     raise ValueError(
        #         f"""Cannot prune to {final_num}; model has size \
        #             {self._weights.size}"""
        #     )
        logging.warn("Using deprecated function `FragModel.prune`")
        atomic_heap: list[FragmentPruneCandidate] = []

        lossvec = numpy.zeros(self.nof_fragments(), dtype=numpy.float64)
        max_size: int = 0

        for graph in tqdm.tqdm(graphs):
            graph_agg: dict[int, list[numpy.float64]] = {}
            prob_agg: list[numpy.float64] = []
            max_size = max(len(graph.get_colors()), max_size)
            all_observations = tuple(self.partition_graph(graph, use_ac))
            graph_labels = frozenset(
                itertools.chain.from_iterable(
                    (match.fragment_id for match in obs.matches)
                    for obs in all_observations
                )
            )
            for observation in all_observations:
                labels_used = frozenset(
                    match.fragment_id for match in observation.matches
                )
                unused_labels = graph_labels - labels_used
                prob_agg.append(observation.weight)
                for label in unused_labels:
                    if label in graph_agg:
                        graph_agg[label].append(observation.weight)
                    else:
                        graph_agg[label] = [observation.weight]

            # calculate total probability for molecule
            total_prob = scipy.special.logsumexp(prob_agg)

            if total_prob == numpy.float64("-inf"):
                raise RuntimeError("Fail")

            partition_agg: dict[int, numpy.float64] = {
                i: scipy.special.logsumexp(v) for i, v in graph_agg.items()
            }

            for label in graph_labels:
                if label not in partition_agg:
                    lossvec[label] += numpy.float64("-inf")
                    continue
                delta = partition_agg[label]
                lossvec[label] += delta - total_prob

        # calculate minimum value for cutoff
        min_cutoff = -numpy.log(max_size * len(graphs))

        # append fragment spec information
        for i in range(len(lossvec)):
            loss_info = self.loglikelihood_frag(i)
            lossvec[i] -= loss_info

        for cgraph, weight, loss in zip(
            (
                self._fragtree.get_fragment(i)
                for i in range(self.nof_fragments())
            ),
            self._weights,
            lossvec,
            strict=True,
        ):
            prune_candidate = FragmentPruneCandidate(
                cgraph.to_canongraph(),
                weight,
                loss,
            )
            if len(prune_candidate.fragment.get_colors()) > 1:
                if prune_candidate.weight == numpy.float64("-inf"):
                    print(
                        "Disposing of useless factor "
                        f"{prune_candidate.fragment}"
                    )
                    continue
                elif prune_candidate.weight < min_cutoff:
                    print(
                        "Disposing of low-occurrence factor "
                        f"{prune_candidate.fragment}"
                    )
                    continue
            heapq.heappush(atomic_heap, prune_candidate)

        final_fragments = []

        for i in range(final_num):
            if len(atomic_heap) == 0:
                continue
            next_item = heapq.heappop(atomic_heap)
            if i > final_num:
                if next_item.is_atomic():
                    raise RuntimeError(
                        "Cannot prune atomic fragments; raise `final_num` "
                        "threshold"
                    )
                break
            final_fragments.append(next_item)

        final_tree = sgrpy.graph.GraphTree.from_pins(
            c.fragment for c in final_fragments
        )

        final_weights = numpy.array(
            [c.weight for c in final_fragments], dtype=numpy.float64
        )

        final_weights -= scipy.special.logsumexp(final_weights)

        return FragModel(_fragtree=final_tree, _weights=final_weights)

    def loglikelihood_frag(self, i: int) -> float:
        logging.warn("Using deprecated function FragModel.loglikelihood_frag")
        num_atomics = self.nof_fragments_atomic()
        fragment = self._fragtree.get_fragment(i).to_canongraph()
        num_components = len(fragment.get_colors())
        num_edges = len(tuple(fragment.get_bonds()))
        component_ll = -num_components * numpy.log(num_atomics)
        edges_ll = -lndfac_unchecked(num_edges)
        return float(component_ll + edges_ll)

    def self_loglikelihood(self) -> float:
        logging.warn("Using deprecated function FragModel.self_loglikelihood")
        return sum(
            self.loglikelihood_frag(i) for i in range(self.nof_fragments())
        )

    def loglikelihood(
        self, graphs: collections.abc.Iterable[sgrpy.graph.CGraph]
    ) -> float:
        logging.warn("Using deprecated function FragModel.loglikelihood")
        logm_list: list[float] = []
        for graph in tqdm.tqdm(graphs):
            logp_list: list[numpy.float64] = []
            for observation in self.partition_graph(graph):
                logp_list.append(observation.weight)
            logm_list.append(float(scipy.special.logsumexp(logp_list)))
        return sum(logm_list)

    def train_em(
        self,
        graphs: collections.abc.Sequence[sgrpy.graph.CGraph],
        n_jobs: int = 1,
        use_ac: bool = False,
    ) -> "FragModel":
        logging.warn("Using deprecated function FragModel.train_em")
        new_weight_agg: list[None | list[numpy.float64]] = [
            None for _ in range(self._weights.size)
        ]

        for graph in tqdm.tqdm(graphs):
            graph_agg: dict[int, list[numpy.float64]] = {}
            prob_agg: list[numpy.float64] = []
            for observation in self.partition_graph(graph, use_ac):
                label_counts = collections.Counter(
                    m.fragment_id for m in observation.matches
                )
                prob_agg.append(observation.weight)

                # tabulate counts and probabilities
                for label, count in label_counts.items():
                    new_value = (
                        numpy.log(numpy.float64(count)) + observation.weight
                    )
                    if label not in graph_agg:
                        graph_agg[label] = [new_value]
                    else:
                        graph_agg[label].append(new_value)

            # calculate total probability for molecule
            total_prob = scipy.special.logsumexp(prob_agg)

            # add values to overall aggregator
            for component, weights in graph_agg.items():
                if total_prob == numpy.float64("-inf"):
                    continue
                new_value = scipy.special.logsumexp(weights) - total_prob
                target_entry = new_weight_agg[component]
                if target_entry is None:
                    new_weight_agg[component] = [new_value]
                else:
                    target_entry.append(new_value)

        # tally all values from aggregator
        new_weights = numpy.asarray(
            [
                numpy.float64("-inf")
                if weightvec is None
                else scipy.special.logsumexp(weightvec)
                for weightvec in new_weight_agg
            ],
            dtype=numpy.float64,
        )

        new_weights -= scipy.special.logsumexp(new_weights)

        return FragModel(_fragtree=self._fragtree, _weights=new_weights)

    @classmethod
    def from_graphs_naive(
        cls,
        graphs: collections.abc.Sequence[sgrpy.graph.CGraph],
        num_fragments: int | None,
        canon_method: sgrpy.graph.CanonType,
        max_size: None | int = None,
        min_count: None | int = None,
        use_bell: bool = False,
    ) -> "FragModel":
        logging.warn("`FragModel.from_graphs_naive` is deprecated")
        obs = tuple(
            bell_naive(
                graphs,
                num_fragments,
                canon_method,
                max_size,
                min_count,
                use_bell,
            )
        )

        final_tree = sgrpy.graph.GraphTree.from_pins(
            p.graph.to_canongraph() for p in obs
        )
        weights = numpy.asarray(
            tuple(p.weight_guess for p in obs), dtype=numpy.float64
        )

        return FragModel(_fragtree=final_tree, _weights=weights)

    def augment_fragments(
        self,
        graphs: collections.abc.Sequence[sgrpy.graph.CGraph],
        min_count: int = 3,
    ) -> "FragModel":
        logging.warn("`FragModel.augment_fragments` is deprecated")
        next_fragment_size: int = 0
        detect_fragments: dict[sgrpy.graph.SCanonGraph, numpy.float64] = {}
        atomic_weights: dict[int, numpy.float64] = {}
        all_fragments: list[FragmentWeighted] = []

        # make list of largest fragments
        for fragment, fweight in (
            (self._fragtree.get_fragment(i), self._weights[i])
            for i in range(self.nof_fragments())
        ):
            all_fragments.append(FragmentWeighted(fragment, fweight))
            cur_fragment_size = len(fragment.to_canongraph().get_colors())
            if cur_fragment_size < next_fragment_size:
                continue
            elif cur_fragment_size == next_fragment_size:
                detect_fragments[fragment] = numpy.float64(fweight)
            else:
                next_fragment_size = cur_fragment_size
                detect_fragments = {fragment: numpy.float64(fweight)}
            if cur_fragment_size == 1:
                atomic_weights[fragment.to_canongraph().get_colors()[0]] = (
                    fweight
                )

        # create fragment tree for detecting fragments
        detect_tree = sgrpy.graph.GraphTree.from_pins(
            scg.to_canongraph() for scg in detect_fragments
        )

        fragment_counts: dict[sgrpy.graph.SCanonGraph, int] = {}
        fragment_weights: dict[sgrpy.graph.SCanonGraph, numpy.float64] = {}

        # locate and tally graph fragments
        for graph in graphs:
            graph_colors = graph.get_colors()
            for locate_map in detect_tree.locate_fragments(graph):
                loc_fragment = detect_tree.get_fragment(locate_map.fragment_id)
                cur_nodes = frozenset(locate_map.mapping.as_tuple_int())
                nodes_map = locate_map.mapping.inv().as_tuple()
                node_neighbors: dict[int, tuple[int, list[int]]] = {}
                for n in cur_nodes:
                    neighbors = frozenset(graph.neighbors(n)).difference(
                        cur_nodes
                    )
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
                    if new_fragment_s in fragment_counts:
                        fragment_counts[new_fragment_s] += 1
                    else:
                        fragment_counts[new_fragment_s] = 1

                    fragment_weight = (
                        numpy.min(
                            [
                                detect_fragments[loc_fragment],
                                atomic_weights[new_color],
                            ],
                        )
                        - 64
                    )

                    if new_fragment_s in fragment_weights:
                        fragment_weights[new_fragment_s] = numpy.min(
                            [fragment_weights[new_fragment_s], fragment_weight]
                        )
                    else:
                        fragment_weights[new_fragment_s] = fragment_weight

        # filter to min_count fragments
        fragment_counts = {
            k: v for k, v in fragment_counts.items() if v >= min_count
        }

        # add fragments to tree
        all_fragments.extend(
            FragmentWeighted(fragment, fragment_weights[fragment])
            for fragment in fragment_counts
        )

        all_fragments.sort(reverse=True)

        fragment_tree = sgrpy.graph.GraphTree.from_pins(
            c.fragment.to_canongraph() for c in all_fragments
        )

        weights = numpy.asarray(
            [c.weight for c in all_fragments], dtype=numpy.float64
        )

        weights -= scipy.special.logsumexp(weights)

        return FragModel(_fragtree=fragment_tree, _weights=weights)

    def partition_graph(
        self, graph: sgrpy.graph.CGraph, use_ac: bool = False
    ) -> collections.abc.Iterable[Observation]:
        logging.warn("`FragModel.partition_graph` is deprecated")
        all_matches = self._fragtree.locate_fragments(graph)
        match_dict = {
            match.mapping.as_tuple_int(): match for match in all_matches
        }

        # check for partial cover
        all_colors = frozenset(itertools.chain(*(match_dict.keys())))
        if all_colors != frozenset(range(len(graph))):
            return
        match_list = list(match_dict.values())
        match_values = [
            -self._weights[match.fragment_id] for match in match_list
        ]
        edge_counts = [
            len(
                tuple(
                    self._fragtree.get_fragment(match.fragment_id)
                    .to_canongraph()
                    .get_bonds()
                )
            )
            for match in match_list
        ]
        vertex_counts = [
            len(
                tuple(
                    self._fragtree.get_fragment(match.fragment_id)
                    .to_canongraph()
                    .get_colors()
                )
            )
            for match in match_list
        ]
        vertex_total = len(tuple(graph.get_colors()))
        edge_total = len(tuple(graph.get_bonds()))
        labels = [match.fragment_id for match in match_list]

        get_top: collections.abc.Iterable[list[int]]
        if use_ac:
            get_top = sgrpy.dl_rs.get_topn_sc(
                list(match_dict.keys()),
                match_values,
                edge_counts,
                edge_total,
                labels,
                vertex_total,
                64,
            )
        else:
            get_top = sgrpy.dl_rs.get_topn(
                list(match_dict.keys()),
                match_values,
                64,
            )
        for covers in get_top:
            known_matches = frozenset(match_list[x] for x in covers)
            frag_counts = numpy.zeros(self._weights.size, dtype=numpy.uintp)
            for match in known_matches:
                frag_counts[match.fragment_id] += 1
            weight_vec = self._weights.copy()
            weight_vec[
                (self._weights == numpy.float64("-inf"))
                & (frag_counts == numpy.uintp(0))
            ] = numpy.float64(0)
            loglikelihood = numpy.dot(weight_vec, frag_counts)
            vertices = [vertex_counts[p] for p in covers]
            local_labels = tuple(
                collections.Counter(labels[p] for p in covers).values()
            )
            ac_factor = calc_ac_factor(
                local_labels,
                vertex_total - sum(vertices),
            )
            edges = [edge_counts[p] for p in covers]
            as_factor = calc_as_factor(edges, edge_total)
            total_weight = loglikelihood - ac_factor - as_factor
            yield Observation(matches=known_matches, weight=total_weight)

    @classmethod
    def from_graphs(
        cls,
        graphs: collections.abc.Iterable[sgrpy.graph.CanonGraph],
        weights: numpy.typing.NDArray[numpy.float64],
    ) -> "FragModel":
        logging.warn(
            "`FragModel.from_graphs` is deprecated. Please use `FragModel.new` "
            "instead"
        )
        fragtree = sgrpy.graph.GraphTree.from_pins(graphs)
        return FragModel(_fragtree=fragtree, _weights=weights)

    def to_json(self) -> JSON:
        weights_list = [float(x) for x in self._weights]
        fragments = tuple(scg.as_str() for scg in self._fragtree.iter_pins())
        return {"weights": weights_list, "fragments": fragments}

    @classmethod
    def from_json(cls, data: JSON) -> "FragModel":
        json_dict = json_to_dict(data)
        weights_list = json_to_list(json_dict["weights"])
        weights = numpy.asarray(weights_list, dtype=numpy.float64)
        fragments_list = json_to_list(json_dict["fragments"])
        fragments_gen = (
            sgrpy.graph.SCanonGraph.from_str(
                str(fs), checked=False
            ).to_canongraph()
            for fs in fragments_list
        )
        tree = sgrpy.graph.GraphTree.from_pins(fragments_gen)
        return FragModel(_fragtree=tree, _weights=weights)
