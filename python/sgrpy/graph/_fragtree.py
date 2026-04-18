"""Fragment lookup tree."""

import dataclasses
import enum
from collections.abc import Iterable
from typing import Any

from sgrpy.iotypes import JSON

from ._canon_types import CanonType
from ._cgraph import CGraph
from ._cgraph_canon import CanonGraph, SCanonGraph
from ._operations import AddNode, SubNode
from ._permutations import IndexMapping


def _conv_opt_int(x: None | int) -> int:
    if x is None:
        return 0
    elif x >= 0:
        return x + 1
    return x


def _conv_int_opt(x: int) -> None | int:
    if x == 0:
        return None
    elif x > 0:
        return x - 1
    return x


@dataclasses.dataclass(frozen=True, slots=True)
class GraphMatch:
    """Graph match datastructure.

    Attributes
    ----------
    fragment_id : int
        Index of the pinned fragment which was found.
    mapping : IndexMapping
        Mapping from the fragment indices to discovered subgraph.
    """

    fragment_id: int
    mapping: IndexMapping


@dataclasses.dataclass(eq=True, frozen=True, slots=True)
class GraphIncQuery:
    from_id: int
    operation: AddNode

    @classmethod
    def from_json(cls, json: JSON) -> "GraphIncQuery":
        jsond: Any = json
        from_id = int(jsond["fid"])
        operation = AddNode(
            connections=tuple(int(i) for i in jsond["op"]["ids"]),
            color=int(jsond["op"]["color"]),
        )
        return GraphIncQuery(from_id=from_id, operation=operation)

    def to_json(self) -> JSON:
        return {
            "fid": self.from_id,
            "op": {
                "color": self.operation.color,
                "ids": list(self.operation.connections),
            },
        }


@dataclasses.dataclass(eq=True, frozen=True, slots=True)
class GraphIncResult:
    to_id: int
    mapping: IndexMapping
    pinned: None | int

    @classmethod
    def from_json(cls, json: JSON) -> "GraphIncResult":
        jsond: Any = json
        to_id = int(jsond["to_id"])
        pinned = _conv_int_opt(int(jsond["pinned"]))
        mapping = IndexMapping.from_seq(
            tuple(_conv_int_opt(int(x)) for x in jsond["mapping"])
        )
        return GraphIncResult(to_id=to_id, mapping=mapping, pinned=pinned)

    def to_json(self) -> JSON:
        return {
            "to_id": self.to_id,
            "pinned": _conv_opt_int(self.pinned),
            "mapping": list(_conv_opt_int(x) for x in self.mapping.as_tuple()),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class FragmentStackEntry:
    current_graph_index: int
    current_cagraph: CanonGraph


class NodeStatus(enum.IntEnum):
    CORE = enum.auto()
    FREE = enum.auto()
    FRBD = enum.auto()


@dataclasses.dataclass(frozen=True, slots=True)
class SearchStackEntry:
    node_state: tuple[NodeStatus | tuple[int, ...], ...]
    cur_prim: int
    cur_map: IndexMapping


@dataclasses.dataclass(frozen=True, slots=True)
class PrimitiveStatus:
    prim_id: int
    pinned: None | int

    @classmethod
    def from_json(cls, json: JSON) -> "PrimitiveStatus":
        jsond: Any = json
        prim_id = int(jsond["prim_id"])
        pinned = _conv_int_opt(int(jsond["pinned"]))
        return PrimitiveStatus(prim_id=prim_id, pinned=pinned)

    def to_json(self) -> JSON:
        return {"prim_id": self.prim_id, "pinned": _conv_opt_int(self.pinned)}


@dataclasses.dataclass(frozen=True, slots=True)
class GraphTree:
    """Tree-based structure containing target fragments to be searched for."""

    _pinned_list: tuple[SCanonGraph, ...]
    _primitives_list: tuple[SCanonGraph, ...]
    _primitives_lookup: dict[SCanonGraph, PrimitiveStatus]
    _fragment_map: dict[GraphIncQuery, GraphIncResult]
    _empty_loc: int

    @classmethod
    def from_pins(
        cls,
        graphs: Iterable[CanonGraph],
    ) -> "GraphTree":
        """Initialize GraphTree from an iterable of pinned fragments.

        These fragments must be canonical (ideally using same method) so that
        their uniqueness may be validated.

        Parameters
        ----------
        graphs : Iterable[CanonGraph]
            Canonical graph fragments to be searched for.

        Returns
        -------
        GraphTree
        """
        # initialize stack and other data structures
        primitives_lookup: dict[SCanonGraph, PrimitiveStatus] = {}
        primitives_list: list[SCanonGraph] = []
        fragment_map: dict[GraphIncQuery, GraphIncResult] = {}
        stack: list[FragmentStackEntry] = []
        empty_graph_loc: None | int = None
        n_p = 0
        for g in graphs:
            if g.cgraph.nof_components() > 1:
                raise NotImplementedError(
                    """Graphs with more than 1 component cannot be used at this
                    time."""
                )
            scg = g.to_scanongraph()
            if scg in primitives_list:
                continue
            primitives_list.append(scg)
            primitives_lookup[scg] = PrimitiveStatus(n_p, n_p)
            stack.append(
                FragmentStackEntry(
                    current_graph_index=n_p,
                    current_cagraph=g,
                )
            )
            n_p += 1

        pinned_list = tuple(primitives_list)

        # begin loop
        while len(stack) > 0:
            # retrieve state variables
            state = stack.pop()
            cur_graph = state.current_cagraph
            cur_index = state.current_graph_index

            # if null graph, continue
            if len(cur_graph) == 0:
                empty_graph_loc = cur_index
                continue

            cur_graph_s = cur_graph.to_scanongraph()
            is_pinned = primitives_lookup[cur_graph_s].pinned

            # determine parents of cur_graph
            for remove_i in range(len(cur_graph)):
                operation = SubNode(remove_i)
                par_graph, c_to_p = cur_graph.sub_node(operation)
                if par_graph.cgraph.nof_components() > 1:
                    continue
                par_graph_s = par_graph.to_scanongraph()
                par_index: int
                if par_graph_s in primitives_lookup:
                    par_index = primitives_lookup[par_graph_s].prim_id
                else:
                    par_index = len(primitives_list)
                    primitives_list.append(par_graph_s)
                    primitives_lookup[par_graph_s] = PrimitiveStatus(
                        par_index,
                        None,
                    )
                    stack.append(
                        FragmentStackEntry(
                            current_graph_index=par_index,
                            current_cagraph=par_graph,
                        )
                    )

                # create mapping query entry
                c_to_p_extend = c_to_p.append(remove_i)
                p_to_c_tuple = c_to_p_extend.inv().as_tuple_int()
                cur_neighbors = cur_graph.cgraph.neighbors(remove_i)
                par_neighbors = tuple((p_to_c_tuple[j] for j in cur_neighbors))
                add_color = cur_graph.get_colors()[remove_i]
                add_operation = AddNode(
                    connections=par_neighbors, color=add_color
                )
                query = GraphIncQuery(
                    from_id=par_index, operation=add_operation
                )

                # create mapping result entry
                result = GraphIncResult(
                    to_id=cur_index,
                    mapping=c_to_p_extend,
                    pinned=is_pinned,
                )

                # # validate
                # test_parent = (
                #     primitives_list[par_index]
                #     .to_canongraph()
                #     .add_node(add_operation)[0]
                #     .to_scanongraph()
                # )
                # test_child = primitives_list[cur_index]
                # assert test_parent == test_child

                fragment_map[query] = result

        if empty_graph_loc is None:
            pinned_list = ()
            empty_node = CanonGraph.empty(CanonType.F)[0].to_scanongraph()
            primitives_list = [empty_node]
            primitives_lookup = {}
            fragment_map = {}
            empty_graph_loc = 0

        return GraphTree(
            _pinned_list=pinned_list,
            _primitives_list=tuple(primitives_list),
            _primitives_lookup=primitives_lookup,
            _fragment_map=fragment_map,
            _empty_loc=empty_graph_loc,
        )

    @classmethod
    def from_json(cls, json: JSON, checked: bool) -> "GraphTree":
        """Initialize GraphTree from a JSON.

        The format of the JSON is not checked even if `checked` is `True`.

        Parameters
        ----------
        json : JSON
            Input data for creation of GraphTree.
        checked : bool
            If True, check canonicity of inputted fragments (recommended if
            JSON has unknown provenance).

        Returns
        -------
        GraphTree
        """
        jsond: Any = json
        pinned_list = tuple(
            SCanonGraph.from_str(str(x), checked) for x in jsond["pinlist"]
        )
        primitives_list = tuple(
            SCanonGraph.from_str(str(x), checked) for x in jsond["primlist"]
        )
        primitives_lookup = {
            SCanonGraph.from_str(
                str(x["k"]), checked
            ): PrimitiveStatus.from_json(x["v"])
            for x in jsond["prim_lookup"]
        }
        fragment_map = {
            GraphIncQuery.from_json(x["k"]): GraphIncResult.from_json(x["v"])
            for x in jsond["fragment_map"]
        }
        empty_loc = int(jsond["empty_loc"])
        return GraphTree(
            _pinned_list=pinned_list,
            _primitives_list=primitives_list,
            _primitives_lookup=primitives_lookup,
            _fragment_map=fragment_map,
            _empty_loc=empty_loc,
        )

    def to_json(self) -> JSON:
        """Convert to JSON form.

        Returns
        -------
        JSON
        """
        pinlist = [x.as_str() for x in self._pinned_list]
        primlist = [x.as_str() for x in self._primitives_list]
        prim_lookup = [
            {"k": k.as_str(), "v": v.to_json()}
            for k, v in self._primitives_lookup.items()
        ]
        fragment_map = [
            {"k": k.to_json(), "v": v.to_json()}
            for k, v in self._fragment_map.items()
        ]
        empty_loc = self._empty_loc
        return {
            "pinlist": pinlist,
            "primlist": primlist,
            "prim_lookup": prim_lookup,
            "fragment_map": fragment_map,
            "empty_loc": empty_loc,
        }

    def nof_pinned(self) -> int:
        """Get number of pinned fragments."""
        return len(self._pinned_list)

    def nof_primitives(self) -> int:
        """Get number of primitive fragments (total fragments in tree)."""
        return len(self._primitives_list)

    def nof_maps(self) -> int:
        """Get number of mappings in tree.

        Returns
        -------
        int
        """
        return len(self._fragment_map)

    def get_fragment(self, id: int) -> SCanonGraph:
        """Get pinned fragment by its index.

        Parameters
        ----------
        id : int
            Index of pinned fragment.

        Returns
        -------
        SCanonGraph
        """
        return self._primitives_list[id]

    def locate_fragments(self, graph: CGraph) -> Iterable[GraphMatch]:
        """Locate pinned fragments within an arbitrary colored graph.

        Parameters
        ----------
        graph : CGraph
            Graph to be searched.

        Returns
        -------
        Iterable[GraphMatch]
        """
        if graph.nof_components() > 1:
            raise NotImplementedError(
                "Matching for unconnected graphs not implemented"
            )

        # initial data structures
        init_node_state: list[NodeStatus] = list(
            NodeStatus.FREE for _ in range(len(graph))
        )
        init_map = IndexMapping.from_seq([None for _ in range(len(graph))])
        init_prim = self._empty_loc
        stack: list[SearchStackEntry] = []
        graph_colors = graph.get_colors()

        # sort_vals = ((i, c) for i, c in enumerate(graph_colors))
        sort_vals = sorted(
            ((i, c) for i, c in enumerate(graph_colors)),
            key=lambda x: len(graph.neighbors(x[0])),
            reverse=True,
        )

        # initialize stack with atomics
        for cur_index, cur_color in sort_vals:
            # create new node state vector, mark visited as forbidden in future
            # also add current to CORE for all new node states
            if (
                init_node_state[cur_index] == NodeStatus.FRBD
                or init_node_state[cur_index] == NodeStatus.CORE
            ):
                continue
            temp_node_state: list[NodeStatus | tuple[int, ...]] = list(
                init_node_state
            )
            temp_node_state[cur_index] = NodeStatus.CORE
            init_node_state[cur_index] = NodeStatus.FRBD

            # get new fragment id
            cur_op = AddNode(tuple(), cur_color)
            cur_query = GraphIncQuery(from_id=init_prim, operation=cur_op)
            cur_result = self._fragment_map.get(cur_query)
            if cur_result is None:
                continue
            new_index = cur_result.to_id

            # set new mapping (trivial for atomics)
            cur_map = init_map.subst(cur_index, 0)

            # if new fragment is pinned, return GraphMatch
            if cur_result.pinned is not None:
                pin_id = cur_result.pinned
                new_match = GraphMatch(
                    fragment_id=pin_id, mapping=cur_map.inv()
                )
                yield new_match

            # find neighbors, mark as ADJC if CORE
            for adj_index in graph.neighbors(cur_index):
                node_state = temp_node_state[adj_index]
                if node_state == NodeStatus.FREE:
                    temp_node_state[adj_index] = (cur_index,)
                elif isinstance(node_state, tuple):
                    temp_node_state[adj_index] = tuple(
                        sorted(node_state + (cur_index,))
                    )
            new_node_state_t = tuple(temp_node_state)

            # create new state and append to stack
            new_state = SearchStackEntry(
                node_state=new_node_state_t, cur_prim=new_index, cur_map=cur_map
            )
            # print(f"New State: {new_state}")
            stack.append(new_state)

        while len(stack) > 0:
            # get current state from stack
            cur_state = stack.pop()
            # print(f"Current State: {cur_state}")
            cur_node_set = cur_state.node_state
            cur_node_state = list(cur_state.node_state)
            cur_prim = cur_state.cur_prim
            cur_map = cur_state.cur_map

            cur_map_tuple = cur_map.as_tuple()

            # begin looping over adjacent nodes
            # next_nodes = sorted(
            #     (
            #         (i, ns)
            #         for i, ns in enumerate(cur_node_set)
            #         if isinstance(ns, tuple)
            #     ),
            #     key=lambda x: (
            #         sum(
            #             1
            #             for i in graph.neighbors(x[0])
            #             if cur_node_set[i] != NodeStatus.FRBD
            #         ),
            #         len(x[1]),
            #         x,
            #     ),
            #     reverse=True,
            # )
            next_nodes = (
                (i, ns)
                for i, ns in enumerate(cur_node_set)
                if isinstance(ns, tuple)
            )

            for new_node_index, new_node_adj in next_nodes:
                # create new node state vector, mark visited as forbidden in
                # future, also add current to CORE for all new node states
                if (
                    cur_node_state[new_node_index] == NodeStatus.FRBD
                    or cur_node_state[new_node_index] == NodeStatus.CORE
                ):
                    continue
                new_node_state = list(cur_node_state)
                new_node_state[new_node_index] = NodeStatus.CORE
                cur_node_state[new_node_index] = NodeStatus.FRBD

                # get new fragment id
                cur_color = graph_colors[new_node_index]
                cur_op_connections = tuple(
                    sorted(
                        d
                        for d in (cur_map_tuple[x] for x in new_node_adj)
                        if d is not None
                    )
                )
                cur_op = AddNode(
                    cur_op_connections,
                    cur_color,
                )
                cur_query = GraphIncQuery(from_id=cur_prim, operation=cur_op)
                cur_result = self._fragment_map.get(cur_query)
                if cur_result is None:
                    continue
                new_prim = cur_result.to_id

                # set new mapping
                new_map = cur_map.subst(
                    new_node_index, len(cur_result.mapping) - 1
                ).compose(cur_result.mapping)

                # if new fragment is pinned, return GraphMatch
                if cur_result.pinned is not None:
                    pin_id = cur_result.pinned
                    new_match = GraphMatch(
                        fragment_id=pin_id, mapping=new_map.inv()
                    )
                    yield new_match

                # find neighbors, mark as ADJC if FREE
                for adj_index in graph.neighbors(new_node_index):
                    node_state = new_node_state[adj_index]
                    if node_state == NodeStatus.FREE:
                        new_node_state[adj_index] = (new_node_index,)
                    elif isinstance(node_state, tuple):
                        new_node_state[adj_index] = tuple(
                            sorted(node_state + (new_node_index,))
                        )
                new_node_state_t = tuple(new_node_state)

                # create new state and append to stack
                new_state = SearchStackEntry(
                    node_state=new_node_state_t,
                    cur_prim=new_prim,
                    cur_map=new_map,
                )
                # print(f"New State: {new_state}")
                stack.append(new_state)

    def iter_pins(self) -> Iterable[SCanonGraph]:
        """Iterate through pinned fragments.

        Returns
        -------
        Iterable[SCanonGraph]
        """
        return self._pinned_list
