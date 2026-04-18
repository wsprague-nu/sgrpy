"""DataFrame I/O helper functions."""

import enum
import pathlib

import polars


class FileType(enum.Enum):
    """Variants of column-based files for import/export."""

    CSV = enum.auto()
    """Comma-separated variable (.csv default)."""
    TSV = enum.auto()
    """Tab-separated variable (.tsv default)."""
    PARQUET = enum.auto()
    """Parquet (.parquet default)."""

    @classmethod
    def from_str(cls, filetype: str) -> "FileType":
        """Initialize filetype from given extension.

        Parameters
        ----------
        filetype : str
            File extension (e.g., ".csv")

        Returns
        -------
        FileType
        """
        match filetype.strip().lstrip(".").lower():
            case "csv":
                return FileType.CSV
            case "tsv":
                return FileType.TSV
            case "parquet":
                return FileType.PARQUET
            case _:
                raise ValueError(
                    f'Unrecognized file format keyword "{filetype}"'
                )

    @classmethod
    def infer(cls, path: pathlib.Path | str) -> "FileType":
        """Infer filetype from given path.

        Parameters
        ----------
        path : pathlib.Path | str
            Path to given file.

        Returns
        -------
        FileType
        """
        if isinstance(path, str):
            path = pathlib.Path(path)
        suffix_orig = path.suffix
        if suffix_orig == "":
            raise ValueError(
                f'Unable to infer file format for "{str(path)}" (no suffix).'
            )
        suffix = suffix_orig.lower()
        match suffix:
            case ".csv":
                return FileType.CSV
            case ".tsv":
                return FileType.TSV
            case ".parquet" | ".pqt":
                return FileType.PARQUET
            case _:
                raise ValueError(
                    f'Unable to infer file format for "{str(path)}" '
                    f'(unrecognized suffix "{suffix}").'
                )


def load_dataframe(
    path: pathlib.Path | str,
    filetype: FileType | None = None,
) -> polars.LazyFrame:
    """Load dataframe from path using `polars`.

    Parameters
    ----------
    path : pathlib.Path | str
        Path to file.
    filetype : FileType | None
        Type of file (default: attempt to infer from suffix).

    Returns
    -------
    polars.LazyFrame
        Lazy dataframe containing information from file.
    """
    if filetype is None:
        filetype = FileType.infer(path)
    match filetype:
        case FileType.CSV:
            return polars.read_csv(path).lazy()
        case FileType.TSV:
            return polars.read_csv(path, separator="\t").lazy()
        case FileType.PARQUET:
            return polars.read_parquet(path).lazy()


def save_dataframe(
    df: polars.LazyFrame | polars.DataFrame,
    path: pathlib.Path | str,
    filetype: FileType | None = None,
) -> None:
    """Save dataframe to path.

    Parameters
    ----------
    df : polars.LazyFrame | polars.DataFrme
        Data to be saved.  If saving to CSV or TSV, must contain only
        basic datatypes (no lists, arrays, or structs).
    path : pathlib.Path | str
        Path to file.
    filetype : FileType | None
        Type of file (default: attempt to infer from suffix).
    """
    if filetype is None:
        filetype = FileType.infer(path)
    lf = df.lazy()
    match filetype:
        case FileType.CSV:
            return lf.sink_csv(path)
        case FileType.TSV:
            return lf.sink_csv(path, separator="\t")
        case FileType.PARQUET:
            return lf.sink_parquet(path)
