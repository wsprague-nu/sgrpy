"""Bond struct."""

import dataclasses


@dataclasses.dataclass(frozen=True, order=True, slots=True)
class Bond:
    """Bond between two nodes; should be created via Bond.new()."""

    _y: int
    _x: int

    @classmethod
    def new(cls, src: int, trg: int) -> "Bond":
        """Create new undirected `Bond` given two indices.

        Bond indices may swap to ensure that `Bond.src` is less than `Bond.trg`.

        Parameters
        ----------
        src : int
            Index of source node.
        trg : int
            Index of target node.

        Returns
        -------
        sgrpy.graph.Bond
        """
        if src < trg:
            return Bond(_x=src, _y=trg)
        return Bond(_x=trg, _y=src)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.new(src={self.src},trg={self.trg})"

    @property
    def src(self) -> int:
        """int: Index of source node."""
        return self._x

    @property
    def trg(self) -> int:
        """int: Index of target node."""
        return self._y

    def as_tuple(self) -> tuple[int, int]:
        """Return `Bond` in tuple form.

        Returns
        -------
        tuple[int,int]
        """
        return (self.src, self.trg)


@dataclasses.dataclass(frozen=True, order=True, slots=True)
class TBond[_T]:
    """Bond between two nodes; should be created via Bond.new()."""

    _y: int
    _x: int
    _c: _T

    @classmethod
    def new(cls, src: int, trg: int, color: _T) -> "TBond[_T]":
        """Create new undirected `Bond` given two indices and a color.

        Parameters
        ----------
        src : int
            Index of source node.
        trg : int
            Index of target node.
        color : _T
            Color of bond.

        Returns
        -------
        sgrpy.graph.TBond[_T]
        """
        if src == trg:
            raise ValueError("Loop bonds are invalid.")
        if src < trg:
            return TBond(_x=src, _y=trg, _c=color)
        return TBond(_x=trg, _y=src, _c=color)

    def canon(self) -> "TBond[_T]":
        """Return canonized bond, obtained by ensuring `src` < `trg`."""
        if self._x < self._y:
            return self
        return dataclasses.replace(self, _x=self._y, _y=self._x)

    @classmethod
    def from_tuple(cls, tup: tuple[int, int, _T]) -> "TBond[_T]":
        """Create new undirected `Bond` given two indices and a color.

        Parameters
        ----------
        tup : tuple[int, int, _T]
            Index of source node.
        trg : int
            Index of target node.
        color : _T
            Color of bond.

        Returns
        -------
        sgrpy.graph.TBond[_T]
        """
        return TBond.new(tup[0], tup[1], tup[2])

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f".new(src={self.src},trg={self.trg},color={self.color})"
        )

    @property
    def src(self) -> int:
        """int: Index of source node."""
        return self._x

    @property
    def trg(self) -> int:
        """int: Index of target node."""
        return self._y

    @property
    def color(self) -> _T:
        """_T: Color property of bond."""
        return self._c

    def as_tuple(self) -> tuple[int, int, _T]:
        """Return `TBond` in tuple form.

        Returns
        -------
        tuple[int,int,_T]
        """
        return (self.src, self.trg, self.color)

    def with_color[_U](self, color: _U) -> "TBond[_U]":
        new_bond: TBond[_U] = dataclasses.replace(self, _c=color)  # type: ignore[arg-type, assignment]
        return new_bond
