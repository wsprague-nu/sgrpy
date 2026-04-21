# SGR-Py (SubGraph Regularization)

This is a Python module which performs molecule encoding, linear feature regression, and subgraph regularization.

## About the project

This project provides various information-based graph routines (including the titular subgraph regularization algorithm, implemented in Rust), linear algebra routines, and an `RDKit`-based shim which translates molecules and SMILES strings into labeled graphs via several possible methods.

## Getting started

### Installation

#### Installation from PyPI

1. Enter a virtual environment, if you want to use one.

2. Run `pip install sgrpy`, or `pip install sgrpy[rdkit]` if you want to include the optional RDKit shim.

#### Installation from Wheel

1. Navigate to relevant Tag or Release (right sidebar).

2. Download relevant `.whl` file for your system.

3. Enter a virtual environment, if you want to use one.

4. Run `pip install rdkit` to install RDKit if you want to use the optional RDKit shim.

5. Run `pip install sgrpy-*.whl` on the wheel file you downloaded to install the main package.

#### Installation from Source

##### Windows

1. Install [Visual Studio 2017](https://visualstudio.microsoft.com/vs/) or later (or Build Tools for Visual Studio with Visual C++ option)

2. Install [Rust toolchain](https://rust-lang.org/tools/install/)

3. Install [Python](https://www.python.org/downloads/) and [RDKit](https://pypi.org/project/rdkit-pypi/) if you want to use the RDKit shim.

4. Enter a virtual environment, if you want to use one.

5. Clone this repo, then install using `pip install .` in the top-level directory.

##### MacOS/Linux

1. Install [Rust toolchain](https://rust-lang.org/tools/install/)

2. Install [Python](https://www.python.org/downloads/) and [RDKit](https://pypi.org/project/rdkit-pypi/) if you want to use the RDKit shim.

3. Enter a virtual environment, if you want to use one.

4. Clone this repo, then install using `pip install .` in the top-level directory.

### Tutorials

- [CLI Tutorial](https://github.com/wsprague-nu/sgrpy/blob/main/doc/source/pages/CLI%20Tutorial.md)
- Python API Tutorial (to be released)

## Contact

William Sprague (main developer) - [wsprague@u.northwestern.edu](mailto:wsprague@u.northwestern.edu)

Project Link: https://github.com/wsprague-nu/sgrpy
