"""Fragment search and storage methods."""

import dataclasses
import enum
from collections.abc import Iterable
from typing import final

from sgrpy.iotypes import JSON

from ._canon_types import CanonType
from ._cbgraph import CBGraph
from ._cbgraph_canon import CBGraphC, CBGraphCS
from ._operations import AddNodeT, Extension
from ._permutations import IndexMapping


@final
@dataclasses.dataclass(frozen=True, slots=True)
class FragmentMatch:
    """Fragment match datastructure.

    fragment_id : int
        Index of the pinned fragment which was found.
    mapping : IndexMapping
        Mapping from the fragment indices to discovered subgraph.
    """

    fragment_id: int
    mapping: IndexMapping


@final
@dataclasses.dataclass(frozen=True, slots=True)
class _GraphIncQuery:
    """Graph increment (addition) operation from specific fragment."""

    from_id: int
    operation: AddNodeT[int, int]

    @classmethod
    def from_json(cls, json: JSON) -> "_GraphIncQuery":
        from_id: int = json["fid"]  # type: ignore[assignment,call-overload,index]
        vert_color: int = json["vt"]  # type: ignore[assignment,call-overload,index]
        operation = AddNodeT.new(
            ext=(
                Extension(idx=int(i), typ=int(ec))
                for i, ec in zip(json["i"], json["ec"], strict=True)  # type: ignore[arg-type,call-overload,index]
            ),
            typ=vert_color,
        )
        query = _GraphIncQuery(from_id=from_id, operation=operation)
        return query

    def to_json(self) -> JSON:
        data_out: dict[str, int | list[int]] = {
            "fid": self.from_id,
            "vt": self.operation.typ,
            "i": [ext.idx for ext in self.operation.ext],
            "ec": [ext.typ for ext in self.operation.ext],
        }
        return data_out


@final
@dataclasses.dataclass(frozen=True, slots=True)
class _GraphIncResult:
    """Complementary datastructure to a `_GraphIncQuery`."""

    to_id: int
    mapping: IndexMapping
    pinned: None | int

    @classmethod
    def from_json(cls, json: JSON) -> "_GraphIncResult":
        to_id: int = json["tid"]  # type: ignore[assignment,index,call-overload]
        pinned: int | None = json["pin"]  # type: ignore[assignment,index,call-overload]
        map_list: list[int | None] = json["map"]  # type: ignore[assignment,index,call-overload]
        mapping = IndexMapping.from_seq(map_list)
        return _GraphIncResult(to_id=to_id, mapping=mapping, pinned=pinned)

    def to_json(self) -> JSON:
        return {
            "tid": self.to_id,
            "pinned": self.pinned,
            "mapping": self.mapping.as_tuple(),
        }


@final
@dataclasses.dataclass(frozen=True, slots=True)
class _FragmentStackEntry:
    """Entry in fragment enumeration (for building tree)."""

    cur_graph_index: int
    cur_fragment: CBGraphC


@final
class _NodeStatus(enum.IntEnum):
    """Status of fragment node."""

    CORE = enum.auto()
    FREE = enum.auto()
    FRBD = enum.auto()


@final
@dataclasses.dataclass(frozen=True, slots=True)
class _SearchStackEntry:
    """Entry in search stack (for subgraph search)."""

    node_state: tuple[_NodeStatus | tuple[Extension[int], ...], ...]
    cur_prim: int
    cur_map: IndexMapping


@final
@dataclasses.dataclass(frozen=True, slots=True)
class _PrimStatus:
    """Status of fragment, including id and whether or not primitive."""

    prim_id: int
    pinned: None | int

    @classmethod
    def from_json(cls, json: JSON) -> "_PrimStatus":
        prim_id: int = json["pid"]  # type: ignore[assignment,index,call-overload]
        pinned: int | None = json["pin"]  # type: ignore[assignment,index,call-overload]
        return _PrimStatus(prim_id=prim_id, pinned=pinned)

    def to_json(self) -> JSON:
        data = {"pid": self.prim_id, "pin": self.pinned}
        return data


@final
@dataclasses.dataclass(frozen=True, slots=True)
class FragmentTree:
    """Tree-based structure capable of subgraph search.

    Caches a fixed set of (canonical) subgraphs and searches for multiple
    matches simultaneously.
    """

    _pinned_list: tuple[CBGraphCS, ...]
    _primitives_list: tuple[CBGraphCS, ...]
    _primitives_lookup: dict[CBGraphCS, _PrimStatus]
    _fragment_map: dict[_GraphIncQuery, _GraphIncResult]
    _empty_loc: int

    @classmethod
    def from_json(cls, json: JSON) -> "FragmentTree":
        raise NotImplementedError

    def to_json(self) -> JSON:
        raise NotImplementedError

    @classmethod
    def from_pins(
        cls, graphs: Iterable[CBGraphC], check_validity: bool = True
    ) -> "FragmentTree":
        """Initialize FragmentTree from the set of searchable subgraphs.

        These fragments must be non-empty, have a single connected component,
        and use the same canonization method.  These properties are checked by
        default unless the `check_validity` argument is used.  This primarily
        increases speed when connectedness does not have to be verified.

        Parameters
        ----------
        graphs : Iterable[CBGraphC]
            Canonical graph fragments to be searched for using the
            `FragmentTree`.
        check_validity : bool
            Ensure properties specified above (default: True).

        Returns
        -------
        FragmentTree
        """
        # initialize core datastructures
        primitives_lookup: dict[CBGraphCS, _PrimStatus] = {}
        primitives_list: list[CBGraphCS] = []
        fragment_map: dict[_GraphIncQuery, _GraphIncResult] = {}
        empty_graph_loc: None | int = None

        # initialize search stack and variables
        stack: list[_FragmentStackEntry] = []
        canon_method: CanonType | None = None
        n_p = 0

        # begin iteration through input, populating initial stack
        for g in graphs:
            # check validity of `g` if enabled
            if check_validity:
                if canon_method is None:
                    canon_method = g.method
                elif canon_method != g.method:
                    raise ValueError("Must use consistent canonization method")
                elif g.nof_nodes() == 0:
                    raise ValueError("Cannot pass empty graph")
                elif g.cbgraph.nof_components() > 1:
                    raise ValueError(
                        "Graphs must have a single connected component"
                    )

            # obtain string-form of graph (for caching purposes)
            scg = g.to_sform()

            # if graph is a repeat of an already-added graph, continue
            if scg in primitives_list:
                continue

            # add entries to primitive list and lookup
            # note: address in pinned list will be the same as in primitive list
            primitives_list.append(scg)
            primitives_lookup[scg] = _PrimStatus(n_p, n_p)

            # add entry to stack to later search for all parents
            stack.append(
                _FragmentStackEntry(cur_graph_index=n_p, cur_fragment=g)
            )

            # increment current pin index
            n_p += 1

        # initialize pinned list (indices same as prim list for now)
        pinned_list = tuple(primitives_list)

        # begin loop searching for all fragment parents
        # note: each fragment will only be visited once due to a uniqueness
        # check inside of the loop
        while len(stack) > 0:
            # retrieve state variables
            state = stack.pop()
            cur_graph = state.cur_fragment
            cur_index = state.cur_graph_index

            # if null graph, continue
            if cur_graph.nof_nodes() == 0:
                empty_graph_loc = cur_index
                continue

            # check whether the current graph is pinned
            cur_graph_s = cur_graph.to_sform()
            is_pinned = primitives_lookup[cur_graph_s].pinned

            # iterate over parents of cur_graph
            for par_graph, c_to_p, add_op, sub_op in cur_graph.get_parents():
                # if parent graph is disconnected, do not proceed
                if par_graph.cbgraph.nof_components() > 1:
                    continue

                # obtain index if parent currently exists, add it if not
                par_graph_s = par_graph.to_sform()
                par_index: int
                if par_graph_s in primitives_lookup:
                    par_index = primitives_lookup[par_graph_s].prim_id
                else:
                    par_index = len(primitives_list)
                    primitives_list.append(par_graph_s)
                    primitives_lookup[par_graph_s] = _PrimStatus(
                        par_index, None
                    )
                    stack.append(
                        _FragmentStackEntry(
                            cur_graph_index=par_index, cur_fragment=par_graph
                        )
                    )

                # create mapping for query entry (ensuring add op is canonical)
                c_to_p_extend = c_to_p.append(sub_op.index)
                query = _GraphIncQuery(
                    from_id=par_index, operation=add_op.sort()
                )

                # create mapping result entry
                result = _GraphIncResult(
                    to_id=cur_index, mapping=c_to_p_extend, pinned=is_pinned
                )

                fragment_map[query] = result

        # ensure empty graph location is not None
        assert empty_graph_loc is not None

        return FragmentTree(
            _pinned_list=pinned_list,
            _primitives_list=tuple(primitives_list),
            _primitives_lookup=primitives_lookup,
            _fragment_map=fragment_map,
            _empty_loc=empty_graph_loc,
        )

    def nof_pinned(self) -> int:
        """Get number of pinned fragments (searchable subgraphs)."""
        return len(self._pinned_list)

    def nof_primitives(self) -> int:
        """Get number of primitive fragments (total subgraphs)."""
        return len(self._primitives_list)

    def nof_maps(self) -> int:
        """Get number of mappings in tree."""
        return len(self._fragment_map)

    def get_fragment(self, frag_id: int) -> CBGraphCS:
        """Get pinned fragment by its index.

        Parameters
        ----------
        frag_id : int
            Index of pinned fragment.

        Returns
        -------
        CBGraphS
        """
        return self._primitives_list[frag_id]

    def locate_fragments(self, graph: CBGraph) -> Iterable[FragmentMatch]:
        """Locate pinned fragments within an arbitrary graph.

        Parameters
        ----------
        graph : CBGraph
            Graph to be searched.

        Returns
        -------
        Iterable[FragmentMatch]
        """
        # initial data structures for building stack
        total_nodes = graph.nof_nodes()
        init_node_state: list[_NodeStatus] = [_NodeStatus.FREE] * total_nodes
        init_map = IndexMapping.from_seq([None] * total_nodes)
        init_prim = self._empty_loc
        stack: list[_SearchStackEntry] = []

        # initialize stack with atomics
        for cur_index, cur_color in enumerate(graph.get_colors()):
            # ignore current node if it is already visited
            if init_node_state[cur_index] in (
                _NodeStatus.CORE,
                _NodeStatus.FRBD,
            ):
                continue

            # create new node state vector, mark visited as forbidden in future
            # also add current to CORE for all new node states
            temp_node_state: list[_NodeStatus | tuple[Extension[int], ...]] = (
                list(init_node_state)
            )
            temp_node_state[cur_index] = _NodeStatus.CORE
            init_node_state[cur_index] = _NodeStatus.FRBD

            # build new query
            cur_op: AddNodeT[int, int] = AddNodeT.new((), typ=cur_color)
            cur_query = _GraphIncQuery(from_id=init_prim, operation=cur_op)

            # check results against fragment tree
            cur_result = self._fragment_map.get(cur_query)
            if cur_result is None:
                # atomic fragment is not present in search tree
                continue
            new_index = cur_result.to_id

            # set new mapping (trivial for atomics)
            cur_map = init_map.subst(cur_index, 0)

            # if new fragment is pinned, yield FragmentMatch
            if cur_result.pinned is not None:
                pin_id = cur_result.pinned
                new_match = FragmentMatch(
                    fragment_id=pin_id, mapping=cur_map.inv()
                )
                yield new_match

            # find neighbors of neighbor, convert to adjacency if CORE
            for adj_ext in graph.neighbors_ext(cur_index):
                node_state = temp_node_state[adj_ext.idx]
                rev_ext = dataclasses.replace(adj_ext, idx=cur_index)
                if node_state == _NodeStatus.FREE:
                    temp_node_state[adj_ext.idx] = (rev_ext,)
                elif isinstance(node_state, tuple):
                    # TODO: CHECK THIS TO MAKE SURE IT DOESN'T NEED SORTING
                    temp_node_state[adj_ext.idx] = node_state + (rev_ext,)

            # create new state and append to stack
            new_state = _SearchStackEntry(
                node_state=tuple(temp_node_state),
                cur_prim=new_index,
                cur_map=cur_map,
            )
            stack.append(new_state)

        # obtain graph colors for later lookups
        graph_colors = graph.get_colors()

        # begin growing fragments and returning matches
        while len(stack) > 0:
            # get current state from stack
            cur_state = stack.pop()
            cur_node_set = cur_state.node_state
            cur_node_state = list(cur_state.node_state)
            cur_prim = cur_state.cur_prim
            cur_map = cur_state.cur_map

            cur_map_tuple = cur_map.as_tuple()

            # get neighbor nodes from current state set
            next_nodes = (
                (i, ns)
                for i, ns in enumerate(cur_node_set)
                if isinstance(ns, tuple)
            )

            # iterate over possible increments
            for new_node_index, new_node_adj in next_nodes:
                # if new node is forbidden or part of CORE, ignore
                if (
                    cur_node_state[new_node_index] == _NodeStatus.FRBD
                    or cur_node_state[new_node_index] == _NodeStatus.CORE
                ):
                    continue

                # create new node state vector, mark visited node as forbidden
                # for future searches, also add current node to CORE for new
                # node states
                new_node_state = list(cur_node_state)
                new_node_state[new_node_index] = _NodeStatus.CORE
                cur_node_state[new_node_index] = _NodeStatus.FRBD

                # get new fragment id
                cur_color = graph_colors[new_node_index]
                cur_op_ext = (
                    dataclasses.replace(x, idx=d)
                    for x, d in (
                        (x, cur_map_tuple[x.idx]) for x in new_node_adj
                    )
                    if d is not None
                )
                cur_op = AddNodeT.new_sort(ext=cur_op_ext, typ=cur_color)

                # build query and check fragment tree for a child fragment
                cur_query = _GraphIncQuery(from_id=cur_prim, operation=cur_op)
                cur_result = self._fragment_map.get(cur_query)
                if cur_result is None:
                    continue
                new_prim = cur_result.to_id

                # set new mapping
                new_map = cur_map.subst(
                    new_node_index, len(cur_result.mapping) - 1
                ).compose(cur_result.mapping)

                # if new fragment is pinned, return FragmentMatch
                if cur_result.pinned is not None:
                    pin_id = cur_result.pinned
                    new_match = FragmentMatch(
                        fragment_id=pin_id, mapping=new_map.inv()
                    )
                    yield new_match

                # find neighbors, mark with adj tuple if FREE
                for adj_ext in graph.neighbors_ext(new_node_index):
                    node_state = new_node_state[adj_ext.idx]
                    rev_ext = dataclasses.replace(adj_ext, idx=new_node_index)
                    if node_state == _NodeStatus.FREE:
                        new_node_state[adj_ext.idx] = (rev_ext,)
                    elif isinstance(node_state, tuple):
                        new_node_state[adj_ext.idx] = node_state + (rev_ext,)

                # create new state and append to stack
                new_state = _SearchStackEntry(
                    node_state=tuple(new_node_state),
                    cur_prim=new_prim,
                    cur_map=new_map,
                )
                stack.append(new_state)

    def iter_pins(self) -> Iterable[CBGraphCS]:
        """Iterate through pinned fragments.

        Returns
        -------
        Iterable[CBGraphCS]
        """
        return self._pinned_list
